---
permalink: neural_style
---
{%- capture customjs -%}
  {%- include_relative _custom.js -%}
{%- endcapture -%}
{%- include_relative _ekko-lightbox.min.js -%}
{{- customjs | uglify -}}
