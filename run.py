
import subprocess
import logging
import os
import json
import sys, argparse
import getpass, MySQLdb
import numpy as np
import pandas as pd

import shutil
import nltk
nltk.download('punkt')

paths = ["./logs_finals/","./images_finals/","./models_finals/"]

# for path in paths:
#     if os.path.exists(path):
#         shutil.rmtree(path)


scripts = {
    "dlatk_path":"./og_dlatk/dlatk/dlatkInterface.py",
    "useful_scripts_path":"./usefulScripts",
    "log_file_path":"./logs_finals/runs.xlsx"
}

working_dir = "/home/dmohanty/L-Factor/"
db = "LFactor2023"
user = getpass.getuser()
# conn = MySQLdb.connect(read_default_file="~/.my.cnf", db=db, user=user)
# cur = conn.cursor()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
# single table at a time
class Run():
    def __init__(self, run_id, params, default_params, scripts_location, save_to_file=True, save_to_file_location = "./models_finals"):
        self.run_id = run_id
        self.dlatk_details = scripts_location
        self.params = dotdict(params)
        self.default_params = dotdict(default_params)
        self.save_to_file = save_to_file
        self.save_to_file_location = f"{save_to_file_location}/{self.run_id}/"
        self.initialize_directory()
        
        self.get_arguments()
        self.table = self.params.table
        self.commands = []

        self.log_file = f'logs_finals/{self.run_id}/{self.table}-testing.log'
        self.initialize_logger()
        
    def initialize_directory(self):
        os.makedirs(f"./models_10k/{self.run_id}/", exist_ok=True)
        os.makedirs(f"./logs_10k/{self.run_id}/", exist_ok=True)
        os.makedirs(f"./images_10k/", exist_ok=True)
        
        os.makedirs(self.save_to_file_location, exist_ok=True)
    
    def initialize_logger(self):
        # logging.info = logging.info.getLogger('basic_logger')
        try:
            os.remove(f"logs_finals/{self.run_id}/{self.table}-testing.log")
        except:
            print("New Table! No previous log file in system")
            # os.makedirs(f"logs_finals/", exist_ok=True)
            os.makedirs(f"logs_finals/{self.run_id}/", exist_ok=True)
        logging.basicConfig(format='%(asctime)s - %(message)s', filename=self.log_file, level=logging.INFO, force = True)
        logging.info("Logger Initialized!")
        logging.info(f"Run for table {self.table} with params {self.params}") 
    
    
    def get_arguments(self):
        for name, value in self.default_params.items():
            if name not in self.params.keys():
                self.params[name] = self.default_params[name]
        
        
        self.params.p_occ_mod = str(self.params.p_occ).replace(".","_") 
        self.params.pmi_thershold_mod = str(self.params.pmi_thershold).replace(".","_") 

        if self.params.feature_types == "ngrams":
            temp = "-n"
            for i in range(1, self.params.n_grams_end+1):
                temp+=f" {i}"
            self.params.combine_feat_tables = f"{self.params.n_grams_start}to{self.params.n_grams_end}"
            self.params.feature = f"feat${self.params.combine_feat_tables}gram${self.params.table}${self.params.column}"
            self.params.add_ngrams = temp
            self.params.suffix = "gram"
            

            self.params.restricted_feature = f"{self.params.feature}"
            self.params.lexicon = f"{self.params.prefix}_{self.params.table}_{self.params.model}_{self.params.combine_feat_tables}{self.params.suffix}"
            self.params.groupfeatures = f"feat$cat_{self.params.lexicon}_w${self.params.table}${self.params.column}${self.params.combine_feat_tables}"
            self.params.groupfeatures_reduced = f"{self.params.prefix}_{self.params.column}_{self.params.table}_{self.params.combine_feat_tables}{self.params.suffix}_{self.params.model}"

            if self.params.feat_occ_filter == True:
                self.params.groupfeatures = f"feat$cat_{self.params.lexicon}{self.params.p_occ_mod}_w${self.params.table}${self.params.column}${self.params.combine_feat_tables}"
                self.params.restricted_feature = f"{self.params.restricted_feature}${self.params.p_occ_mod}"
                self.params.lexicon = f"{self.params.lexicon}{self.params.p_occ_mod}"
                self.params.groupfeatures_reduced = f"{self.params.groupfeatures_reduced}_{self.params.p_occ_mod}"

            if self.params.feat_colloc_filter == True:
                self.params.restricted_feature = f"{self.params.restricted_feature}$pmi{self.params.pmi_thershold_mod}"
                # self.params.lexicon = f"{self.params.lexicon}{self.params.p_occ_mod}"
                # self.params.groupfeatures_reduced = f"{self.params.groupfeatures_reduced}_{self.params.p_occ_mod}"

        elif self.params.feature_types == "embeddings":
            # feat$bert_ba_un_meL11L12sun$ibel$new_user_id
            for name in ["embedding_layers","embedding_layer_aggregation","embedding_msg_aggregation"]:
                if name not in self.params.keys():
                    self.params[name] = embedding_model_defaults[self.params.embedding_model][name]
            temp = self.params.embedding_model.split("-")
            feat = temp[0]

            # emb model specs
            if len(temp) > 1:
                for i in range(1, len(temp[1:])+1):
                    feat = feat + "_" + temp[i][:2]
            else:
                feat = feat + "_"
            # message aggregator
            feat = feat + "_" + self.params.embedding_msg_aggregation[:2]

            # layers
            for i in self.params.embedding_layers.split():
                feat = feat + f"L{i}"

            feat = feat + f"{self.params.embedding_layer_aggregation[:2]}n"
            
            self.params.combine_feat_tables = feat
            self.params.feature = f"feat${feat}${self.params.table}${self.params.column}"
            self.params.suffix = ""
            self.params.p_occ_mod = ""
            self.params.restricted_feature = f"{self.params.feature}"
            self.params.lexicon = f"{self.params.prefix}_{self.params.table}_{self.params.model}_{temp[0]}_{self.run_id}"
            # feat$cat_LF1_ibel_pca_bert_w$ibel$new_user_id$bert 
            temp[0] = temp[0][:4]
            self.params.groupfeatures = f"feat$cat_{self.params.lexicon}_w${self.params.table}${self.params.column}${temp[0]}"
            self.params.groupfeatures_reduced = f"{self.params.prefix}_{self.params.column}_{self.params.table}_{self.params.combine_feat_tables}{self.params.suffix}_{self.params.model}"

        # elif self.params.feature_types == "lda":

        os.makedirs(f"images_finals/{self.run_id}/{self.params.table}/", exist_ok=True)
        # self.lexicon = f"{self.prefix}_{self.table}_{self.model}_{self.combine_feat_tables}{self.p_occ}".replace("0.","")
        self.params.save_model_name = f"{self.save_to_file_location}/{self.params.table}_{self.params.model}_{self.params.combine_feat_tables}{self.params.suffix}{self.params.p_occ_mod}.pickle"
        self.params.save_wc_location = f"images_finals/{self.run_id}/{self.params.table}"
        
    def run_command(self, command):
        # command_str = " ".join(command)
        print(f"Executing command: {command}\n")
        self.commands.append(command)
        try:
            logging.info(f"Executing command: {command}")    
            # with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=working_dir, shell=True) as result:
            #     # for line in result.stderr:
            #     #     logging.info(line)
            #     while True:
            #         # Use read1() instead of read() or Popen.communicate() as both blocks until EOF
            #         # https://docs.python.org/3/library/io.html#io.BufferedIOBase.read1
            #         # text = result.stdout.read1().decode("utf-8")
            #         # logging.info(text)
            #         # print(text, end='', flush=True)
            #         output = result.stderr.readline()
            #         if (output == '') and (result.poll() is not None):
            #             break
            #         if output:
            #             logging.info(output.strip())
            #     rc = result.poll()
            if not testing:
                result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=working_dir, shell=True)
            # # with result.stderr:
            # #     for line in iter(process.stderr.readline, b''):
            # #         print(line.decode("utf-8").strip())
                logging.info(f"Command executed successfully for table {self.table}:")
                logging.info(f"STDOUT: {result.stdout.decode('utf-8')}")
                logging.info(f"STDERR: {result.stderr.decode('utf-8')}")        
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing command for table {self.table}:")
            print(e.stderr.decode('utf-8'))
    
    def create_ngrams(self):
        command = [
            self.dlatk_details["dlatk_path"],
            "-d", self.params.database,
            "-t", self.params.table,
            "-g", self.params.column,
            "--message_field", self.params.message_field,
            "--messageid_field", self.params.messageid_field,
            "--add_ngrams", self.params.add_ngrams,
            "--combine_feat_tables", f"{self.params.combine_feat_tables}gram"
        ]
        command = " ".join(command)
        logging.info("[PROCESS] Creating Ngrams!")    
        return self.run_command(command)

    def restrict_ngrams(self):
        command = [
        self.dlatk_details["dlatk_path"],
        "-d", self.params.database,
        "-t", self.params.table,
        "-g", self.params.column,
        "--message_field", self.params.message_field,
        "--messageid_field", self.params.messageid_field,
         "--group_freq_thresh", str(self.params.group_freq_thresh),
        "-f", f"'{self.params.feature}'"
        ]
        
        if self.params.feat_occ_filter:
            command += [f"--feat_occ_filter --set_p_occ {str(self.params.p_occ)}"]

        if self.params.feat_colloc_filter:
            command += [f"--feat_colloc_filter --set_pmi_threshold {str(self.params.pmi_thershold)}"]
             
        command = " ".join(command)
        logging.info("[PROCESS] Restrictings Ngrams!")    
        return self.run_command(command)

    def run_pca(self):
        command = [
            self.dlatk_details["dlatk_path"],
            "-d", self.params.database,
            "-t", self.params.table,
            "-c", self.params.column,
            "--group_freq_thresh", str(self.params.group_freq_thresh),
            "-f", f"'{self.params.restricted_feature}'",
            "--fit_reducer",
            "--model", self.params.model,
            "--n_components", str(self.params.n_components),
            "--reducer_to_lexicon", self.params.lexicon
        ]
        if self.save_to_file:
            command += [f"--save_model --picklefile {self.params.save_model_name}"]
        # print(command)
        command = " ".join(command)
        
        logging.info(f"[PROCESS] Running {self.params.model}!")
        self.run_command(command)
    
    def add_group_scores(self):
        command = [
            self.dlatk_details["dlatk_path"],
           "-d", self.params.database,
            "-t", self.params.table,
            "-c", self.params.column,
            "--group_freq_thresh", str(self.params.group_freq_thresh),
            "--word_table", f"'{self.params.restricted_feature}'",
            "-l", self.params.lexicon,
            "--add_lex_table",
            "--weighted_lex"
        ]
        command = " ".join(command)
        logging.info("[PROCESS] Adding Lexicon!")
        self.run_command(command)
    
    
    def get_data(table, db, scripts, path_to_store="./rawdata"):
        os.makedirs(path_to_store, exist_ok=True)
        sql = f'{scripts["useful_scripts_path"]}/mysqlToCSV.bash {db} "select * from {table}" > {path_to_store}/{table}.csv'
        print(sql, os.listdir())
        run_command(sql, table)

    # def run_command_in_sql(queries):
    #     conn = MySQLdb.connect(read_default_file="~/.my.cnf", db=db, user=user)
    #     cur = conn.cursor()
    #     for query in queries:
    #         query = query.replace('\n', '\n\t')
    #         print(f"[SQL COMMAND]: {query}\n")
    #         try:
    #             cur.execute(query)
    #             print(cur.fetchall())
    #         except Exception as e:
    #             print(e)
    #             pass
    #     conn.close()
        # return cur.fetchall()
    
    def subset_group_scores(self):
        logging.info("[PROCESS] Getting User Scores!")
        command = [f"""DROP TABLE IF EXISTS {self.params.groupfeatures_reduced};"""]
        command += [f"""CREATE TABLE {self.params.groupfeatures_reduced} select group_id as {self.params.column}, group_norm as COMPONENT_0 from {self.params.groupfeatures}  where feat = 'component_0';"""]
        run_command_in_sql(command)

    def generate_general_word_clouds(self):
        command = [
                self.dlatk_details["dlatk_path"],
               "-d", self.params.database,
                "-t", self.params.table,
                "-c", self.params.column,
                "-f", f"'{self.params.restricted_feature}'",
                "--group_freq_thresh", str(self.params.group_freq_thresh),
                "--outcome_table", self.params.groupfeatures_reduced,
                "--outcomes", self.params.wc_component,
                "--tagcloud --tagcloud_colorscheme bluered --rmatrix --csv --make_wordclouds  --output_name", self.params.save_wc_location
            ]
        command = " ".join(command)
        logging.info("[PROCESS] Creating General WordClouds!")
        self.run_command(command)


    def generate_loading_word_clouds(self):
        command = [
                self.dlatk_details["dlatk_path"],
               "-d", self.params.database,
                "-t", self.params.table,
                "-c", self.params.column,
                "--make_all_topic_wordclouds --num_topic_words", str(self.params.wc_num_topic_words),
                "--topic_lex", self.params.lexicon,
                "--output_name", f"{self.params.save_wc_location}_loadings"
            ]
        command = " ".join(command)
        logging.info("[PROCESS] Creating Loadings WordClouds!")
        self.run_command(command)

    def create_embeddings(self):
        command = [
                self.dlatk_details["dlatk_path"],
               "-d", self.params.database,
                "-t", self.params.table,
                "-c", self.params.column,
                "--message_field", self.params.message_field,
                "--messageid_field", self.params.messageid_field,
                "--group_freq_thresh", str(self.params.group_freq_thresh),
                "--add_emb_feat --emb_model", self.params.embedding_model,
                "--embedding_layers", str(self.params.embedding_layers),
                "--embedding_layer_aggregation", str(self.params.embedding_layer_aggregation),
                "--embedding_msg_aggregation",str(self.params.embedding_msg_aggregation)
            ]
        command = " ".join(command)
        logging.info("[PROCESS] Creating Embeddings!")
        self.run_command(command)
        
    def process(self):
        if self.params.feature_types == 'ngrams':
            self.create_ngrams()
            self.restrict_ngrams()
        elif self.params.feature_types == 'embeddings':
            self.create_embeddings()
        else:
            print(f"Set ngrams or embeddings to True")
            NotImplementedError
        
        self.run_pca()
        self.add_group_scores()
        self.subset_group_scores()
        self.generate_general_word_clouds()
        self.generate_loading_word_clouds()
        self.add_to_run_logs_df()

    def add_to_run_logs_df(self):
        self.log_df = pd.DataFrame(columns=['RunID'] + list(run.params.keys()) + ['LogFile','Commands'])
        self.log_df.at[0, "RunID"] = self.run_id
        self.log_df.at[0,  "LogFile"] = self.log_file
        index = len(self.log_df)
        for k, v in run.params.items():
            self.log_df.at[0, k] = f"'{v}'"
        self.log_df.at[0, "Commands"] = self.commands

def run_command_in_sql(queries):
    conn = MySQLdb.connect(read_default_file="~/.my.cnf", db=db, user=user)
    cur = conn.cursor()
    for query in queries:
        logging.info(f"[SQL COMMAND] {query}")
        query = query.replace('\n', '\n\t')
        print(f"[SQL COMMAND]: {query}\n")
        try:
            cur.execute(query)
            print(cur.fetchall())
        except Exception as e:
            print(e)
            pass
    conn.close()
    # return cur.fetchall()

## added 
def check_for_existing_tables(key):
    print(f"[INFO] Checking for existing table: {key}")
    success = False
    try:
        conn = MySQLdb.connect(read_default_file="~/.my.cnf", db=db, user=user)
        new_table = pd.read_sql_query(sql = f"select * from {key} limit 10", con = conn)
        conn.close()
        try:
            assert ("user_id" in list(new_table.columns)), "No user ID column"
            assert ("message" in list(new_table.columns)), "No message column"
            assert ("message_id" in list(new_table.columns)), "No message ID column"
            success = True
        except Exception as e:
            print(f"[ERROR] Required Columns missing ... creating new tables")
            print(f"[ERROR] Exception Details {e}")
    except Exception as e:
        print(f"[ERROR] Checking for existing table failed ... creating new tables")
        print(f"[ERROR] Exception Details {e}")
    return success

## added 
def standardize_tables(config_table, force_create = False, additions = ""):
    tables_to_create = {}
    for _, value in config_table.items():
        
        for k, v in value.items():
            if k == "new_table" and v not in tables_to_create.keys():
                tables_to_create[v] = {}
                tables_to_create[v]["original_table"] = value["original_table"]
                tables_to_create[v]["original_userid"] = value["original_userid"]
                tables_to_create[v]["original_message_field"] = value["original_message_field"]
                tables_to_create[v]["original_messageid_field"] = value["original_messageid_field"]
    
    
    
    for key, val in tables_to_create.items():
        
        command = []
        if (not force_create) & (check_for_existing_tables(key)):
            pass
        else:
            print(f"[INFO] Creating table for {key}, Values {val}")
            conn = MySQLdb.connect(read_default_file="~/.my.cnf", db=db, user=user)
            sample_data = pd.read_sql_query(sql = f"select * from {val['original_table']} limit 10", con = conn)
            conn.close()
            command += [f"""DROP TABLE IF EXISTS {key};"""]
            command += [f"""CREATE TABLE {key} LIKE {val['original_table']};"""]
            filtered_user_table = val['original_table'] + "_filtered_usergroups"
            command += [f"""INSERT INTO {key} SELECT * FROM {val['original_table']} where {val['original_userid']} in (select {val['original_userid']} from {filtered_user_table}) {additions};"""]
        
            # if message doesn't exist
            if "message" not in sample_data.columns:
                command += [
                            f"""ALTER TABLE {key} DROP COLUMN IF EXISTS message;"""
                            f"""ALTER TABLE {key} ADD COLUMN message text;""",
                            f"""UPDATE {key} SET message = {val['original_message_field']};"""]
            
            # if message id doesn't exist
            if "message_id" not in sample_data.columns or val['original_messageid_field'] != "message_id":
                command += [f"""ALTER TABLE {key} DROP COLUMN IF EXISTS message_id;""",
                            f"""ALTER TABLE {key} ADD COLUMN message_id INT;""",
                            f"""SET @row_number = 0;""",
                            f"""UPDATE {key} SET message_id = (@row_number := @row_number + 1) ORDER BY {val['original_messageid_field']};""",
                            f"""ALTER TABLE {key} ADD PRIMARY KEY (message_id);"""]
        
            # if userid doesn't exist
            if "user_id" not in sample_data.columns or val['original_userid'] != "user_id":
                command += [
                    f"""ALTER TABLE {key} DROP COLUMN IF EXISTS user_id;""",
                    f"""ALTER TABLE {key} ADD COLUMN user_id INT;""",
                    f"""DROP TABLE IF EXISTS {key}_usergroups;""",
                    f"""CREATE TABLE {key}_usergroups (user_id INT AUTO_INCREMENT PRIMARY KEY, {val['original_userid']} VARCHAR(30) UNIQUE);""",
                    f"""INSERT INTO {key}_usergroups ({val['original_userid']}) SELECT DISTINCT {val['original_userid']} FROM {key};""",
                    f"""UPDATE {key} n JOIN {key}_usergroups b ON n.{val['original_userid']} = b.{val['original_userid']} SET n.user_id = b.user_id;"""
                ]
            run_command_in_sql(command)
        
        if not check_for_existing_tables(key):
            print("[ERROR] Unable to create table, check logs")
            print("[ERROR] Deleting entries from config_table")
            entries_to_remove = []
            for entry_id, value in config_table.items():
                if value["new_table"] == key:
                    entries_to_remove.append(entry_id)
    
            for entry in entries_to_remove:
                del config_table[entry]
        else:
            for entry_id, value in config_table.items():
                if value["new_table"] == key:
                    config_table[entry_id]["table"] = key
                    config_table[entry_id]["column"] = "user_id"
                    config_table[entry_id]["message_field"] = "message"
                    config_table[entry_id]["messageid_field"] = "message_id"
                    config_table[entry_id]["sql_commands"] = command
    print(f"[INFO] New Config Table {config_table}")
    return config_table

## Setting for testing
testing = False
additions = ""
# if testing:
# additions="limit 5000000"

#### for testing
default_params = {
     "prefix":"LF1",
    "database": "LFactor2023",
    "group_freq_thresh": 500,
    "p_occ": 0.01,
    "feat_occ_filter":True,
    "feat_colloc_filter":True,
    "pmi_thershold": 3.0,
    "feature_types":"ngrams", #"ngrams","lda","embedding"
    "n_grams_start": 1,
    "n_grams_end":3,
    "model": "pca",
    "n_components": 1,
    "wc_component":"COMPONENT_0",
    "wc_num_topic_words":100,
    # "embeddings":False,
    "embedding_model": "bert-base-cased"
}
# suffix = "com2hk"
# suffix = "com1m"
# suffix = "msg"
tables = {
  0: {
    "original_table": "ibelieve_dedup_v1",
    "original_userid": "message_id",
    "original_message_field": "Essay",
    "original_messageid_field": "message_id",   
    "new_table" : f"ibel",
    "group_freq_thresh": 50
  },
2: {
    "original_table": "nytimes_dedupe_v1",
    "original_userid": "message_id",
    "original_message_field": "Body",
    "original_messageid_field": "message_id",   
    "new_table" : f"nyt",
    # "group_freq_thresh": 100
    "p_occ":0.05,
    # "":
  },
3: {
    "original_table": "candor_transcript_dedup",
    "original_userid": "group_id",
    "original_message_field": "utterance",
    "original_messageid_field": "message_id",   
    "new_table" : f"cndr",
     "group_freq_thresh": 1000,
     "p_occ":0.05
  },
    1: {
    "original_table": "GuiltyKnowledge_SOC",
    "original_userid": "message_id",
    "original_message_field": "SOC_Task",
    "original_messageid_field": "message_id",   
    "new_table" : f"socmsg",
    "group_freq_thresh": 5,
    # "feat_occ_filter":True,
    # "p_occ":0.02,
    # "feat_colloc_filter":False,
    # "wc_num_topic_words":100,
  },
4: {
    "original_table": "askreddit_comments_15_20_en",
    "original_userid": "user_id",
    "original_message_field": "message",
    "original_messageid_field": "message_id",   
    "new_table" : f"rddt",
    "group_freq_thresh":1000,
    "p_occ":0.05
    },
5: {
    "original_table": "fb20_dedup",
    "original_userid": "user_id",
    "original_message_field": "message",
    "original_messageid_field": "message_id",
    "new_table" : f"fb",
    "group_freq_thresh":1000,
    "p_occ":0.05
  },
6: {
    "original_table": "twitter19_20_dedup_en",
    "original_userid": "user_id",
    "original_message_field": "message",
    "original_messageid_field": "message_id",   
    "new_table" : f"twtr",
    "group_freq_thresh":1000,
    "p_occ":0.05
  },
    "7": {
    "original_table": "ds4ud_fb_text_posts_p3",
    "original_userid": "fb_user_id",
    "original_message_field": "message",
    "original_messageid_field": "message_id",
    "new_table": "ds4ud",
    "group_freq_thresh": 1000,
    "p_occ":0.05
}
}

# second to last (1) -- done
# last (1)-- done
# second to last; last (2) -- done
# first; second to last; last (3)-- done
# last 4 (4)-- done
# first; middle; second to last; last (4)
# first; last 4 (5)
# first; middle; last 4 (6)

embedding_model_defaults = {
    # "bert-base-cased":{
    #     "embedding_layers": "11",
    #     "embedding_layer_aggregation": "concatenate",
    #     "embedding_msg_aggregation": "mean"
    # },
    "roberta-base":{
        0: {
            "embedding_layers": "11 12",
            "embedding_layer_aggregation": "concatenate",
            "embedding_msg_aggregation": "mean"
        },
        1: {
            "embedding_layers": "12",
            "embedding_layer_aggregation": "concatenate",
            "embedding_msg_aggregation": "mean"
        },
        2: {
            "embedding_layers": "11",
            "embedding_layer_aggregation": "concatenate",
            "embedding_msg_aggregation": "mean"
        },
        # 3: {
        #     "embedding_layers": "1 11 12",
        #     "embedding_layer_aggregation": "concatenate",
        #     "embedding_msg_aggregation": "mean"
        # },
        # 4: {
        #     "embedding_layers": "9 10 11 12",
        #     "embedding_layer_aggregation": "concatenate",
        #     "embedding_msg_aggregation": "mean"
        # },
        # 5: {
        #     "embedding_layers": "1 6 11 12",
        #     "embedding_layer_aggregation": "concatenate",
        #     "embedding_msg_aggregation": "mean"
        # },
        # 5: {
        #     "embedding_layers": "1 9 10 11 12",
        #     "embedding_layer_aggregation": "concatenate",
        #     "embedding_msg_aggregation": "mean"
        # },
        # 6: {
        #     "embedding_layers": "1 6 9 10 11 12",
        #     "embedding_layer_aggregation": "concatenate",
        #     "embedding_msg_aggregation": "mean"
        # }


    },
    # "distilroberta-base":{
    #     "embedding_layers": "11",
    #     "embedding_layer_aggregation": "concatenate",
    #     "embedding_msg_aggregation": "mean"
    # },
    # "xlnet-base-cased":{
    #     "embedding_layers": "11",
    #     "embedding_layer_aggregation": "concatenate",
    #     "embedding_msg_aggregation": "mean"
    # },
    "gpt2":{
        0: {
            "embedding_layers": "10 11",
            "embedding_layer_aggregation": "concatenate",
            "embedding_msg_aggregation": "mean"
        },
        1: {
            "embedding_layers": "11",
            "embedding_layer_aggregation": "concatenate",
            "embedding_msg_aggregation": "mean"
        },
        2: {
            "embedding_layers": "10",
            "embedding_layer_aggregation": "concatenate",
            "embedding_msg_aggregation": "mean"
        },
        # 3: {
        #     "embedding_layers": "1 10 11",
        #     "embedding_layer_aggregation": "concatenate",
        #     "embedding_msg_aggregation": "mean"
        # },
        # 4: {
        #     "embedding_layers": "8 9 10 11",
        #     "embedding_layer_aggregation": "concatenate",
        #     "embedding_msg_aggregation": "mean"
        # },
        # 5: {
        #     "embedding_layers": "1 6 10 11",
        #     "embedding_layer_aggregation": "concatenate",
        #     "embedding_msg_aggregation": "mean"
        # },
        # 5: {
        #     "embedding_layers": "1 8 9 10 11",
        #     "embedding_layer_aggregation": "concatenate",
        #     "embedding_msg_aggregation": "mean"
        # },
        # 6: {
        #     "embedding_layers": "1 6 8 9 10 11",
        #     "embedding_layer_aggregation": "concatenate",
        #     "embedding_msg_aggregation": "mean"
        # }
    }
}

try:
    run_logs_df = pd.read_excel(scripts['log_file_path'])
except Exception as e:
    print(f"[ERROR] {e}")
    run_logs_df = pd.DataFrame()

# ./og_dlatk/dlatk/dlatkInterface.py -d LFactor2023 -t GuiltyKnowledge_SOC -c message_id --message_field SOC_Task --clean_messages
feature_types = ["ngrams"]
config_table = {}
index = 0
for k, table in tables.items():
    for ft in feature_types:
        
        if ft == "ngrams":
            print(f"NGRAMS: {index}")
            config_table[index] = table.copy()
            config_table[index]["feature_types"] = ft
            index+=1
        
        else:
            print("YOLO")


feature_types = ["embeddings"]
# config_table = dict({})
index = len(config_table)
for k, table in tables.items():
    for ft in feature_types:
        for model, param in embedding_model_defaults.items():
            print(index)
            for _, params in param.items():
                config_table[index] = table.copy()
                config_table[index]["feature_types"] = ft
                config_table[index]['embedding_model'] = model
                # print(params)
                for k1, v in params.items():
                    config_table[index][k1] = v
                index+=1


print(f"# of Runs {len(config_table)}")
print(config_table)

config_table = standardize_tables(config_table, additions=additions, force_create=True)



run_id = len(run_logs_df)
print(f"Run Settings (Default): {default_params}")
for index, value in config_table.items():
    print(f"Run for entry #{index}: {value}")
    
    # initialize run
    run = Run(run_id, params=dotdict(value), default_params=default_params, scripts_location=scripts)
    run.process()
    
    # print(run.params)
    print(f"Run for entry #{index} finished!")
    run_logs_df = pd.concat([run_logs_df, run.log_df])
    run_id += 1

run_logs_df.to_excel(scripts['log_file_path'], index=False)
