package main

import (
	"bytes"
	"encoding/json"
	"io"
	"testing"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
)

const testManifest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func TestPublishRequest(t *testing.T) {
	o, err := parseOptions([]string{"--socket", "/work/csi.sock", "--operation", "publish",
		"--volume-id", "trial-target", "--target", "/work/target", "--ref", testManifest}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	r := publishRequest(o)
	if r.GetVolumeId() != "trial-target" || r.GetTargetPath() != "/work/target" || !r.GetReadonly() {
		t.Fatalf("wrong publish ownership or readonly fields: %v", r)
	}
	if r.GetVolumeCapability().GetMount() == nil ||
		r.GetVolumeCapability().GetAccessMode().GetMode() != csipb.VolumeCapability_AccessMode_SINGLE_NODE_WRITER {
		t.Fatalf("wrong mount capability: %v", r.GetVolumeCapability())
	}
	if r.GetVolumeContext()["ref"] != testManifest || r.GetVolumeContext()["csi.storage.k8s.io/ephemeral"] != "true" {
		t.Fatalf("wrong volume context: %v", r.GetVolumeContext())
	}
}

func TestInvalidArgumentsProduceOneFailureReceipt(t *testing.T) {
	base := []string{"--socket", "/work/csi.sock", "--operation", "publish",
		"--volume-id", "trial", "--target", "/work/target", "--ref", testManifest}
	for _, tc := range []struct {
		name string
		args []string
	}{
		{"missing arguments", nil},
		{"relative socket", append(append([]string{}, base...), "--socket", "relative.sock")},
		{"relative target", append(append([]string{}, base...), "--target", "relative")},
		{"empty volume id", append(append([]string{}, base...), "--volume-id", " ")},
		{"unknown operation", append(append([]string{}, base...), "--operation", "acquire")},
		{"ref is mutable", append(append([]string{}, base...), "--ref", "latest")},
		{"bad digest", append(append([]string{}, base...), "--ref", "sha256:xyz")},
		{"zero timeout", append(append([]string{}, base...), "--timeout", "0s")},
		{"extra argument", append(append([]string{}, base...), "extra")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out bytes.Buffer
			if run(tc.args, &out, io.Discard) == 0 {
				t.Fatal("invalid request succeeded")
			}
			var r receipt
			decoder := json.NewDecoder(&out)
			if err := decoder.Decode(&r); err != nil {
				t.Fatal(err)
			}
			if r.Success || r.Error == "" || r.GRPCCode != "InvalidArgument" || r.EndUnixNS < r.StartUnixNS {
				t.Fatalf("invalid failure receipt: %+v", r)
			}
			var extra any
			if err := decoder.Decode(&extra); err != io.EOF {
				t.Fatalf("stdout contains more than one JSON object: %v", err)
			}
		})
	}
}

func TestUnpublishNeedsNoRef(t *testing.T) {
	o, err := parseOptions([]string{"--socket", "/work/csi.sock", "--operation", "unpublish",
		"--volume-id", "trial", "--target", "/work/target"}, io.Discard)
	if err != nil || o.Ref != "" {
		t.Fatalf("unpublish arguments: %+v, %v", o, err)
	}
}
