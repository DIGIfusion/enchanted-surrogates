const fs = require("fs");

function readCoverage(path) {
    if (!fs.existsSync(path)) {
        console.log(`Coverage file not found: ${path}`);
        return {
            total: 0,
            files: {}
        };
    }

    const data = JSON.parse(fs.readFileSync(path, "utf8"));
    const folders = {};

    // Get accumulated coverage for each file in a folder
    for (const [file, info] of Object.entries(data.files)) {
        const folder = path.dirname(file);

        if (!folders[folder]) {
            folders[folder] = {
                covered: 0,
                total: 0
            };
        }

        folders[folder].covered += info.summary.covered_lines;
        folders[folder].total += info.summary.num_statements;
    }

    const coverage = {};

    for (const [folder, stats] of Object.entries(folders)) {
        coverage[folder] = (stats.covered / stats.total) * 100;
    }

    return {
        total: data.totals.percent_covered,
        files: coverage
    };
}

function diffCoverage(base, pr) {
    // All unique file names
    const files = new Set([
        ...Object.keys(base.files),
        ...Object.keys(pr.files)
    ]);

    const result = [];

    for (const file of files) {
        const base_cov = base.files[file] ?? 0;
        const pr_cov = pr.files[file] ?? 0;

        result.push({
            file,
            base: base_cov,
            pr: pr_cov,
            diff: pr_cov - base_cov
        });
    }

    // Create markdown table
    let table = "| File | Line Rate | Change |\n";
    table += "| :--- | :---: | ---: |\n";

    for (const row of result) {
        const sign = row.diff > 0 ? "+" : "";
        const icon =
            row.diff > 0 ? "🟢" :
            row.diff < 0 ? "🔴" :
            "⚪";
        table += `| ${row.file} | ${row.pr.toFixed(1)}% | ${sign}${row.diff.toFixed(1)}% ${icon} |\n`;
    }

    const total_diff = pr.total - base.total;
    const sign = total_diff > 0 ? "+" : "";
    const icon =
        total_diff > 0 ? "🟢" :
        total_diff < 0 ? "🔴" :
        "⚪";
    table += `| **Summary** | **${pr.total.toFixed(1)}%** | **${sign}${total_diff.toFixed(1)}%** ${icon} |\n`;

    return table;
}

module.exports = {
    readCoverage, diffCoverage
};