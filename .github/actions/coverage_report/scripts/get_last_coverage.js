const fs = require('fs');

module.exports = async ({ github, context }) => {
    if (!context.payload.pull_request) {
        console.log("Not a pull request event, exiting.");
        return;
    }

    const { owner, repo } = context.repo;
    const targetBranch = context.payload.pull_request.base.ref;

    const runs = await github.rest.actions.listWorkflowRuns({
        owner,
        repo,
        workflow_id: 'testing.yml',
        branch: targetBranch,
        status: 'success'
    });

    if (!runs.data.workflow_runs || runs.data.workflow_runs.length === 0) {
        console.log("No workflow runs were found for the given criteria");
        return;
    }

    const latestRunId = runs.data.workflow_runs[0].id;
    const artifacts = await github.rest.actions.listWorkflowRunArtifacts({
        owner,
        repo,
        run_id: latestRunId
    });

    if (!runs.data.artifacts || runs.data.artifacts.length === 0) {
        console.log("No artifacts were found for run id " + latestRunId.toString());
        return;
    }

    const coverageArtifact = artifacts.data.artifacts.find(
        artifact => artifact.name === 'combined_coverage'
    );

    if (!coverageArtifact) {
        console.log("Artifact 'combined_coverage' was not found from run id " + latestRunId.toString());
        return;
    }

    //const downloadUrl = coverageArtifact.archive_download_url;
    const download = await github.rest.actions.downloadArtifact({
        owner,
        repo,
        artifact_id: coverageArtifact.id,
        archive_format: "zip"
    });

    fs.writeFileSync("base_coverage.zip", Buffer.from(download.data));

    console.log("Downloaded base branch coverage artifact.");
};