const fs = require('fs');

/**
 * Downloads the latest successful coverage artifact from the base branch
 * of a pull request and saves it locally as `base_coverage.zip`.
 */
module.exports = async ({ github, context }) => {
    if (!context.payload.pull_request) {
        console.log("Not a pull request event, exiting.");
        return;
    }

    const { owner, repo } = context.repo;
    const target_branch = context.payload.pull_request.base.ref;

    const runs = await github.rest.actions.listWorkflowRuns({
        owner,
        repo,
        workflow_id: 'testing.yml',
        branch: target_branch,
        status: 'success'
    });

    if (!runs.data.workflow_runs || runs.data.workflow_runs.length === 0) {
        console.log("No workflow runs were found for the given criteria");
        return;
    }

    const latest_run_id = runs.data.workflow_runs[0].id;
    const artifacts = await github.rest.actions.listWorkflowRunArtifacts({
        owner,
        repo,
        run_id: latest_run_id
    });

    if (!artifacts.data.artifacts || artifacts.data.artifacts.length === 0) {
        console.log("No artifacts were found for run id " + latest_run_id.toString());
        return;
    }

    const coverage_artifact = artifacts.data.artifacts.find(
        artifact => artifact.name === 'combined_coverage'
    );

    if (!coverage_artifact) {
        console.log("Artifact 'combined_coverage' was not found from run id " + latest_run_id.toString());
        return;
    }

    const download = await github.rest.actions.downloadArtifact({
        owner,
        repo,
        artifact_id: coverage_artifact.id,
        archive_format: "zip"
    });

    fs.writeFileSync("base_coverage.zip", Buffer.from(download.data));

    console.log("Downloaded base branch coverage artifact.");
};