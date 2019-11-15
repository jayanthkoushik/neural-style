---
permalink: assets/main
---
{%- capture customjs -%}
  {%- include_relative _custom.js -%}
{%- endcapture -%}
{%- include_relative _jquery-slim.min.js -%}
{%- include_relative _bootstrap.min.js -%}
{%- include_relative _ekko-lightbox.min.js -%}
{%- include_relative _katex.min.js -%}
{%- include_relative _auto-render.min.js -%}
{{- customjs | uglify -}}
