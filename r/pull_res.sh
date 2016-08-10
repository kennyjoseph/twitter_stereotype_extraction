#!/usr/bin/env bash

#DATA_DIR = "foundation:/usr1/kjoseph/thesis_work/lcss_study/python/new_recent"
DATA_DIR="foundation:/usr1/kjoseph/twitter_stereotype_extraction/python/random_user_runs/processed_out/output"
OUT_DIR="."
N_VAL="300"

#cp ../RECENT_RES/all_user_geo_gender.csv ${OUT_DIR}
scp ${DATA_DIR}/\{sent_res_final/${N_VAL}*,index_to*,user_uids.txt,sent_ids_list.txt,sent_res_final/sent_mu_0.npy,assoc_res_final/${N_VAL}*\} ${OUT_DIR}
