const fs = require('fs');
const path = require('path');

const { readCoverage, diffCoverage } = require('./get_coverage_diff');

/**
 * Reads `coverage.json` files from folders `coverage` and `base_coverage`,
 * creates the diff report from those and sends it as a comment on the PR.
 */
module.exports = async ({ github, context }) => {
    const base_coverage = readCoverage("base_coverage/coverage.json");
    const pr_coverage = readCoverage("coverage/coverage.json");

    const diff = diffCoverage(base_coverage, pr_coverage);

    const cov_by_file = fs.readFileSync('coverage/coverage_by_file.txt', 'utf8');

    const body = "## Test Coverage Report\n"
        + diff
        + "\n<details>\n"
        + "<summary>Coverage by file</summary>\n"
        + "\n```text\n"
        + cov_by_file
        + "\n```"
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