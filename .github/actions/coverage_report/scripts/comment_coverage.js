const fs = require('fs');

module.exports = async ({ github, context }) => {
    const coverage = fs.readFileSync('code-coverage-results.md', 'utf8');
    const cov_by_file = fs.readFileSync('coverage/coverage_by_file.txt', 'utf8');

    const body = "## Test Coverage Report\n"
        + coverage
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