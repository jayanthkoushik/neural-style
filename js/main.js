---
permalink: assets/main
---
{%- capture customjs -%}
  {%- include_relative _custom.js -%}
{%- endcapture -%}
{%- include_relative _jquery-slim.min.js -%}
{%- include_relative _bootstrap.min.js -%}
{%- include_relative _ekko-lightbox.min.js -%}
{{- customjs | uglify -}}
