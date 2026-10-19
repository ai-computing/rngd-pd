{{- define "rngd-pd.image" -}}
{{ .Values.image.repository }}{{ with .Values.image.tag }}:{{ . }}{{ end }}
{{- end -}}

{{/* nodeAffinity: the override if set, else "has this accelerator label" */}}
{{- define "rngd-pd.affinity" -}}
{{- $root := index . 0 -}}{{- $label := index . 1 -}}
{{- if $root.Values.nodeAffinity }}
affinity:
  nodeAffinity:
{{ toYaml $root.Values.nodeAffinity | indent 4 }}
{{- else if $label }}
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - {key: {{ $label }}, operator: Exists}
{{- end }}
{{- end -}}

{{- define "rngd-pd.env" -}}
{{- $root := index . 0 -}}{{- $extra := index . 1 -}}
{{- $fixed := dict "PYTHONPATH" $root.Values.pythonPath "PYTHONUNBUFFERED" "1" "VLLM_PLUGINS" "" "MODEL" $root.Values.model "PD_SHM_ROOT" $root.Values.shm.root "AIRM_KV_TRANSPORT" ($root.Values.airm.kvTransport | default "shm") }}
{{- if $root.Values.kvRelay.enabled }}
{{- $_ := set $fixed "AIRM_KV_TCP_ADDR" (printf "%s-kv-relay:%v" $root.Release.Name $root.Values.kvRelay.port) }}
{{- end }}
{{- range $k, $v := merge (dict) $extra $root.Values.env $fixed }}
- {name: {{ $k }}, value: {{ $v | quote }}}
{{- end }}
{{- end -}}

{{- define "rngd-pd.mounts" -}}
{{- range .Values.hostMounts }}
- {name: {{ .name }}, mountPath: {{ .mountPath }}, readOnly: {{ .readOnly | default false }}}
{{- end }}
{{- end -}}

{{- define "rngd-pd.volumes" -}}
{{- range .Values.hostMounts }}
- {name: {{ .name }}, hostPath: {path: {{ .hostPath }}}}
{{- end }}
{{- end -}}

{{- define "rngd-pd.command" -}}
{{- $root := index . 0 -}}
command: [bash, -c]
args:
- |
  set -eu
  {{- with $root.Values.bootstrap }}
  {{- . | nindent 2 }}
  {{- end }}
  exec {{ $root.Values.python }} -m rngd_pd.serving.cluster {{ index . 1 }}
{{- end -}}
