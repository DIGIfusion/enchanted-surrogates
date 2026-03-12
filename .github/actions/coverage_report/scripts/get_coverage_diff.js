const fs = require("fs");

function readCoverage(path) {
    if (!fs.existsSync(path)) {
        console.log(`Coverage file not found: ${path}`);
        return {};
    }

    const data = JSON.parse(fs.readFileSync(path, "utf8"));
    const coverage = {};

    for (const [file, info] of Object.entries(data.files)) {
        coverage[file] = info.summary.percent_covered;
    }

    return coverage;
}

function diffCoverage(base, pr) {
    // All unique file names
    const files = new Set([
        ...Object.keys(base),
        ...Object.keys(pr)
    ]);

    const result = [];

    for (const file of files) {
        const base_cov = base[file] ?? 0;
        const pr_cov = pr[file] ?? 0;

        result.push({
            file,
            base: base_cov,
            pr: pr_cov,
            diff: pr_cov - base_cov
        });
    }

    // Create markdown table
    let table = "| File | PR | Base | Diff |\n";
    table += "|---|---|---|---|\n";

    for (const row of result) {
        const sign = row.diff > 0 ? "+" : "";
        table += `| ${row.file} | ${row.pr.toFixed(1)}% | ${row.base.toFixed(1)}% | ${sign}${row.diff.toFixed(1)}% |\n`;
    }

    return table;
}

module.exports = {
    readCoverage, diffCoverage
};