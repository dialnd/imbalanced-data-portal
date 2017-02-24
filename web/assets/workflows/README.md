# Workflows Directory

This directory is used for temporary storage for active interactive workflows.
Each directory is generated with a hash based on the user and timestamp, and on
logout the temporarily saved data is deleted and the state of the workflow is
saved to the database as a JSON file.
