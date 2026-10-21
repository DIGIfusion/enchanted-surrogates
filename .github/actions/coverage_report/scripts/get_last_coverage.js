const fs = require('fs');

/**
 * Downloads the latest coverage artifact of the base branch of a pull request
 * that has not expired and saves it locally as `base_coverage.zip`.
 *
 * Artifacts are deleted after the retention period, so the newest successful run
 * of the base branch can have an expired one. In that case older runs are tried.
 * If there is none the file is not written and the coverage report says that the
 * base coverage was not available. The comparison is only informational, so
 * nothing here fails the workflow.
 */
module.exports = async ({ github, context }) => {
    if (!context.payload.pull_request) {
        console.log("Not a pull request event, exiting.");
        return;
    }

    const { owner, repo } = context.repo;
    const target_branch = context.payload.pull_request.base.ref;

    try {
        const runs = await github.rest.actions.listWorkflowRuns({
            owner,
            repo,
            workflow_id: 'testing.yml',
            branch: target_branch,
            status: 'success',
            per_page: 20
        });

        const workflow_runs = runs.data.workflow_runs ?? [];
        if (workflow_runs.length === 0) {
            console.log("No workflow runs were found for the given criteria");
            return;
        }

        // Newest run first
        for (const run of workflow_runs) {
            const artifacts = await github.rest.actions.listWorkflowRunArtifacts({
                owner,
                repo,
                run_id: run.id
            });

            const coverage_artifact = (artifacts.data.artifacts ?? []).find(
                artifact => artifact.name === 'combined_coverage' && !artifact.expired
            );

            if (!coverage_artifact) {
                console.log(`Run ${run.id} has no unexpired 'combined_coverage' artifact, trying an older run.`);
                continue;
            }

            try {
                const download = await github.rest.actions.downloadArtifact({
                    owner,
                    repo,
                    artifact_id: coverage_artifact.id,
                    archive_format: "zip"
                });

                fs.writeFileSync("base_coverage.zip", Buffer.from(download.data));
                console.log(`Downloaded base branch coverage artifact from run ${run.id}.`);
                return;
            } catch (error) {
                // For example 410 "Artifact has expired" if it expired after it was listed
                console.log(`Could not download the artifact of run ${run.id} (${error.status ?? "no status"}: ${error.message}), trying an older run.`);
            }
        }

        console.log(`No unexpired coverage artifact was found for the base branch ${target_branch}.`);
    } catch (error) {
        console.log(`Could not fetch the coverage of the base branch (${error.status ?? "no status"}: ${error.message}).`);
    }
};
