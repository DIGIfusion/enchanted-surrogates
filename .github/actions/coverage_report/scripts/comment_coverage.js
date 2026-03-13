const fs = require('fs');
const path = require('path');

const { readCoverage, diffCoverage } = require('./get_coverage_diff');

/**
 * Reads `coverage.json` files from folders `coverage` and `base_coverage`,
 * creates the diff report from those and sends it as a comment on the PR.
 */
module.exports = async ({ github, context }) => {
    const base_coverage = readCoverage("base_coverage/coverage.json", true);
    const pr_coverage = readCoverage("coverage/coverage.json", true);
    const diff = diffCoverage(base_coverage, pr_coverage);

    const base_coverage_by_file = readCoverage("base_coverage/coverage.json", false);
    const pr_coverage_by_file = readCoverage("coverage/coverage.json", false);
    const diff_by_file = diffCoverage(base_coverage_by_file, pr_coverage_by_file);

    const base_name = context.payload.pull_request.base.ref ?? "N/A"
    const branch_text = pr_coverage.total > 0 
        ? "Comparing to coverage data from branch " + base_name 
        : "Coverage data for base branch " + base_name + " was not available"

    const body = "## Test Coverage Report\n"
        + diff
        + "\n_" + branch_text + "_\n"
        + "\n<details>\n"
        + "<summary>Coverage by file</summary>\n"
        + "\n" + diff_by_file + "\n"
        + "\n</details>";

    const { owner, repo } = context.repo;
    const issue_number = context.issue.number;

    const comments = await github.rest.issues.listComments({
        owner,
        repo,
        issue_number
    });

    const botComment = comments.data.find(comment =>
        comment.user.type === 'Bot' &&
        comment.body.includes('Test Coverage Report')
    );

    if (botComment) {
        await github.rest.issues.updateComment({
            owner,
            repo,
            comment_id: botComment.id,
            body
        });
    } else {
        await github.rest.issues.createComment({
            owner,
            repo,
            issue_number,
            body
        });
    }
};