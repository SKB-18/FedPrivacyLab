{{/*
Expand the name of the chart.
*/}}
{{- define "fedprivacylab.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create chart label.
*/}}
{{- define "fedprivacylab.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "fedprivacylab.labels" -}}
helm.sh/chart: {{ include "fedprivacylab.chart" . }}
app.kubernetes.io/name: {{ include "fedprivacylab.name" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}
