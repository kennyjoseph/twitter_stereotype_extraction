# Twitter Stereotype Extraction

This is the github repository for the following paper:

```
@inproceedings{joseph_girls_2017,
	title = {Girls rule, boys drool: Extracting semantic and affective stereotypes on Twitter},
	url = {http://dl.acm.org/citation.cfm?id=2883027},
	booktitle = {Proceedings of the 20th ACM Conference on Computer-supported Cooperative Work and Social Computing},
	author = {Joseph, Kenneth and Wei, Wei and Carley, Kathleen M.},
	year = {2017}
}
```

If you use the code, please let me know so I can help with any issues! And if you don't want to do that, at least cite the paper (please) :)

# Data 

Due to Twitter's TOS, I can't release the raw JSON of the tweets. Instead what I can release is a processed version of it, which is more useful for the present work anyway. The data is located [here]() in a modified Co-NLL format explained below. With the data in this format you'll just be able to run the models and/or experiment with creating new models with this dependency-parsed version of the data. If you want to re-collect the original data, however, the tweet IDs are included, so you're more than welcome to do that!

## Data Collection Strategy

This section describes how I collected the data for this paper - as noted above, however, you *do not* have to run this to replicate the results from the paper - I've already provided the data in the format output from the end of the steps in this section (i.e. after running ```4_run_all_processing_on_ferg_data.py```)

*NOTE:* There are a lot of relative paths and some absolute paths in here. The relative paths will work if you run from the ```data_collection``` directory, the absolute paths you'll have to change.

1. Run ```0_recollect_geotagged_users.py``` with ```geotagged_uid.txt``` as the input data ... this recollected additional data for users geotagged from our original 1.2M sample available [here](https://github.com/kennyjoseph/ferguson_data)
2. Run ```1_get_census_name_data_user.ipynb``` to extract census blocks and user names for users
3. Run ```2_gather_data_for_analysis.ipynb``` ... this combines the old data we had with the new data and then to write out information on the demographics of a user's area and whether or not their name is male or female. This spits out a file that can be used by ```../r/genderize_names.r```
4. Run ```3_train_identity_model.py``` to train an identity extraction model to run on the data
5. Run either ```4_run_all_processing_on_ferg_data.py``` to reproduce the POS tagging, identity extraction and dependency parsing of the data

## Processed data format - modified Co-NLL

This is the same format outputted by the identity extraction tool in the [twitter_dm](https://github.com/kennyjoseph/twitter_dm) library, so just replicating what that readme says.

Each file in the data directory linked to above for this paper is formatted as a modified Co-NLL format - essentially, just the Co-NLL format with some extra information tagged on to the end. An example of two fields of the output for a single tweet are below:
```
1	@theScore	thescore	@	@	penn_treebank_pos=USR	-1	_	_	_	582192856980410368	147290321	03-29-15	O
2	hey	hey	!	!	penn_treebank_pos=UH	0	_	_	_	582192856980410368	147290321	03-29-15	O
```

A single output file will have multiple tweets from multiple users, separated by a newline.  The first eight fields of the output format correspond to the existing CoNLL format described at http://ilk.uvt.nl/conll/#dataformat.

The next fields are tagged on at the end - note that the file ```twitter_dm/identity_extraction/dependency_parse_object.py``` employs the logic used to read these files.

- Field 9: The ID of this tweet
- Field 10: The ID of the user who sent this tweet
- Field 11: The date on which this tweet was sent
- Field 12: The identity label for that term in the tweet - an O is for Outside an identity label (i.e. not an identity label); an I is for Inside an identity label (i.e. part of an identity label)


# Running/Replicating the Model

This code requires Apache Spark and pyspark to work. I ran this stuff on a big machine (64 processors, 512 GB RAM), so on a smaller machine it will probably take a while.

To replicate, first download the processed data from the link above and extract it.  

Then, do the following at the shell:

```
$ mkdir output
$ python 1_get_textunits_for_analysis.py PATH_TO_DATA output/textunits/ 
```

This script processes the data in a format that can then be used to run the model by extracting the sentiment constraints. I'm happy to share this format as well to avoid this intermittent step, again, just let me know.

After that, open up the four ipython notebooks left in the ```python``` directory and run them in succession. They will run the model (```2_run_models.ipynb``` - this takes the longest), generate results for the sentiment baseline model (```3_get_data_for_sentiment_baseline.ipynb```) check convergence of parameters (```4_ck_convergence.ipynb```) and finally run the evaluation (```4_evaluation.ipynb```)

*Note:* The script ```0_gen_empirical_priors_unk.py``` generates the empirical priors for certain identities without survey data from a random 1% sample of the data. I did not include this 1% sample because its too large, I only included the output of this run. Feel free to ping me for the data, though.

# Replicating Results in the Paper

First, download the results we generated by running the model from [here](https://dl.dropboxusercontent.com/u/53207718/NEWEST_RES.zip), place them in the ```r``` directory and unzip that file (or copy the same files over from your run of the model to the same place). Then, running the ```gen_results.R``` script will replicate the results.


