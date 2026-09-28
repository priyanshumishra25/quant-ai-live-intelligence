export function lineChart(svg: SVGSVGElement, series: Array<{values:number[]; className:string}>) {
  svg.innerHTML = "";
  const all = series.flatMap(s => s.values).filter(Number.isFinite);
  if (all.length < 2) return;
  const min = Math.min(...all), max = Math.max(...all), range = max - min || 1;
  for (let i=1;i<5;i++) {
    const y = 10 + i * 44;
    svg.insertAdjacentHTML("beforeend", `<line x1="0" y1="${y}" x2="900" y2="${y}" class="grid-line"/>`);
  }
  series.forEach(s => {
    if (s.values.length < 2) return;
    const points = s.values.map((v,i) => `${(i/(s.values.length-1))*900},${220-((v-min)/range)*190}`).join(" ");
    svg.insertAdjacentHTML("beforeend", `<polyline points="${points}" class="chart-line ${s.className}"/>`);
  });
}
