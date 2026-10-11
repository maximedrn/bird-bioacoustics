<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bird acoustic recognition experiments</title>
  <style>
    @page { size: auto; margin: 0; }
    body {
      color: $foreground;
      background: $background;
      font-family: Arial, sans-serif;
      font-size: 10pt;
      line-height: 1.5;
      margin: 0;
    }
    h1 { font-size: 20pt; line-height: 1.2; }
    h2 { font-size: 14pt; margin-top: 1.5em; }
    h3 { font-size: 11pt; }
    h1, h2, h3 { break-after: avoid; }
    .cell { margin-bottom: 1em; }
    .output_area { break-inside: avoid; }
    .prompt { display: none; }
    .output_area, .output_subarea, .output_scroll {
      max-height: none !important;
      height: auto !important;
      overflow: visible !important;
    }
    table {
      width: 100%;
      table-layout: fixed;
      border-collapse: collapse;
      margin: 0.6em 0 1.2em;
      font-size: 8.5pt;
    }
    th, td {
      overflow-wrap: anywhere;
      white-space: normal !important;
      padding: 0.35em 0.45em;
      vertical-align: top;
      text-align: right;
      border-bottom: 1px solid currentColor;
    }
    th:first-child, td:first-child { text-align: left; }
    caption { text-align: left; font-weight: bold; margin-bottom: 0.5em; }
    img { display: block; max-width: 100%; height: auto; }
    pre { white-space: pre-wrap; }
    code { font-family: monospace; }
    a { color: $link; }
    @media print {
      body { box-sizing: border-box; width: 210mm; padding: 12mm; }
    }
    @media screen {
      body { max-width: 190mm; margin: 24px auto; padding: 0 24px; }
    }
  </style>
</head>
<body>
$body
</body>
</html>
