# Data Collection

NOTE: There are a lot of relative paths and some absolute paths in here. The absolute paths will work if you run from the ```data_collection``` directory, the absolute paths you'll have to change.

1. Run ```0_recollect_geotagged_users.py``` with ```geotagged_uid.txt``` as the input data ... this recollected additional data for users geotagged from our original 1.2M sample
2. Run ```1_get_census_name_data_user.ipynb``` to extract census blocks and user names for users
3. Run ```2_gather_data_for_analysis.ipynb``` ... this combines the old data we had with the new data and then to write out information on the demographics of a user's area and whether or not their name is male or female. This spits out a file that can be used by ```../r/genderize_names.r```
4. Run ```3_train_identity_model.py``` to train an identity extraction model to run on the data
5. Run either ```4_run_all_processing_on_ferg_data.py``` to reproduce.  

Following this, you can run the model by running the four ipython notebooks in this directory starting with NUMBER_ in order. They take about a day, ish, to run through, the first one is probably the longest (about 8 hours)

Note that ```find_best_constraint_model.ipynb``` was used to test models with various kinds of constraints, might be useful for doing so in the future.

