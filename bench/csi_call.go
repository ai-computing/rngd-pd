// Build inside the existing onggi Go module: go build -o csi-call /path/to/csi_call.go.
package main

import (
	"context"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
)

type options struct {
	Socket    string        `json:"socket"`
	Operation string        `json:"operation"`
	VolumeID  string        `json:"volume_id"`
	Target    string        `json:"target"`
	Ref       string        `json:"ref"`
	Timeout   time.Duration `json:"timeout_ns"`
}

type receipt struct {
	options
	RPCMethod   string          `json:"rpc_method"`
	Request     json.RawMessage `json:"request"`
	StartUnixNS int64           `json:"start_unix_ns"`
	EndUnixNS   int64           `json:"end_unix_ns"`
	Success     bool            `json:"success"`
	GRPCCode    string          `json:"grpc_code"`
	Error       string          `json:"error"`
}

func parseOptions(args []string, stderr io.Writer) (options, error) {
	var o options
	f := flag.NewFlagSet("csi-call", flag.ContinueOnError)
	f.SetOutput(stderr)
	f.StringVar(&o.Socket, "socket", "", "absolute Unix socket path")
	f.StringVar(&o.Operation, "operation", "", "publish or unpublish")
	f.StringVar(&o.VolumeID, "volume-id", "", "owned CSI volume ID")
	f.StringVar(&o.Target, "target", "", "absolute owned target path")
	f.StringVar(&o.Ref, "ref", "", "frozen sha256 manifest digest (required for publish)")
	f.DurationVar(&o.Timeout, "timeout", 300*time.Second, "RPC deadline")
	if err := f.Parse(args); err != nil {
		return o, err
	}
	if f.NArg() != 0 {
		return o, fmt.Errorf("unexpected positional arguments")
	}
	if !filepath.IsAbs(o.Socket) || !filepath.IsAbs(o.Target) {
		return o, fmt.Errorf("--socket and --target must be absolute paths")
	}
	if strings.TrimSpace(o.VolumeID) == "" {
		return o, fmt.Errorf("--volume-id is required")
	}
	if o.Operation != "publish" && o.Operation != "unpublish" {
		return o, fmt.Errorf("--operation must be publish or unpublish")
	}
	if o.Timeout <= 0 {
		return o, fmt.Errorf("--timeout must be positive")
	}
	if o.Operation == "publish" {
		if !strings.HasPrefix(o.Ref, "sha256:") || len(o.Ref) != 71 {
			return o, fmt.Errorf("publish requires --ref sha256:<64 hex digits>")
		}
		if _, err := hex.DecodeString(o.Ref[7:]); err != nil {
			return o, fmt.Errorf("invalid manifest digest: %w", err)
		}
	}
	return o, nil
}

func publishRequest(o options) *csipb.NodePublishVolumeRequest {
	return &csipb.NodePublishVolumeRequest{
		VolumeId: o.VolumeID, TargetPath: o.Target, Readonly: true,
		VolumeCapability: &csipb.VolumeCapability{
			AccessType: &csipb.VolumeCapability_Mount{Mount: &csipb.VolumeCapability_MountVolume{}},
			AccessMode: &csipb.VolumeCapability_AccessMode{Mode: csipb.VolumeCapability_AccessMode_SINGLE_NODE_WRITER},
		},
		VolumeContext: map[string]string{"ref": o.Ref, "csi.storage.k8s.io/ephemeral": "true"},
	}
}

func run(args []string, stdout, stderr io.Writer) int {
	r := receipt{StartUnixNS: time.Now().UnixNano(), GRPCCode: codes.InvalidArgument.String()}
	o, err := parseOptions(args, stderr)
	r.options = o
	if err == nil {
		err = call(o, &r)
		r.GRPCCode = status.Code(err).String()
	}
	r.EndUnixNS = time.Now().UnixNano()
	r.Success = err == nil
	if err != nil {
		r.Error = err.Error()
	}
	if encodeErr := json.NewEncoder(stdout).Encode(r); encodeErr != nil {
		fmt.Fprintln(stderr, "receipt:", encodeErr)
		return 1
	}
	if err != nil {
		return 1
	}
	return 0
}

func call(o options, r *receipt) error {
	var request proto.Message
	if o.Operation == "publish" {
		r.RPCMethod = "/csi.v1.Node/NodePublishVolume"
		request = publishRequest(o)
	} else {
		r.RPCMethod = "/csi.v1.Node/NodeUnpublishVolume"
		request = &csipb.NodeUnpublishVolumeRequest{VolumeId: o.VolumeID, TargetPath: o.Target}
	}
	data, err := (protojson.MarshalOptions{UseProtoNames: true}).Marshal(request)
	if err != nil {
		return err
	}
	r.Request = data
	ctx, cancel := context.WithTimeout(context.Background(), o.Timeout)
	defer cancel()
	conn, err := grpc.NewClient("passthrough:///csi",
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", o.Socket)
		}))
	if err != nil {
		return err
	}
	defer conn.Close()
	client := csipb.NewNodeClient(conn)
	if o.Operation == "publish" {
		_, err = client.NodePublishVolume(ctx, request.(*csipb.NodePublishVolumeRequest))
	} else {
		_, err = client.NodeUnpublishVolume(ctx, request.(*csipb.NodeUnpublishVolumeRequest))
	}
	return err
}

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}
