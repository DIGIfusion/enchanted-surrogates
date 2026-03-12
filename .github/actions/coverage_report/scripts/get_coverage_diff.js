const fs = require("fs");
const path = require("path");

/**
 * Extract coverage data by package/folder from the given json file.
 * @param {string} file_path - Path to the coverage json
 * @returns {{ total: number, files: Record<string, number> }}
 *   - total: Overall project coverage as a percentage
 *   - files: Mapping package names to coverage percentages
 */
function readCoverage(file_path) {
    if (!fs.existsSync(file_path)) {
        console.log(`Coverage file not found: ${file_path}`);
        return {
            total: 0,
            files: {}
        };
    }

    const data = JSON.parse(fs.readFileSync(file_path, "utf8"));
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

/**
 * Computes the coverage differences between two coverage objects and generates markdown table.
 * @param {{ total: number, files: Record<string, number> }} base - Coverage data for the base (target) branch.
 * @param {{ total: number, files: Record<string, number> }} pr - Coverage data for the pull request branch.
 * @returns {string} Markdown-formatted table showing coverage per folder/package and diff
 */
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

    // Sort packages by biggest change on top
    result.sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff));

    // Create markdown table
    let table = "| Package | Line Rate | Change |\n";
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