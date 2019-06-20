import click
import json
from tqdm import tqdm_notebook

@click.group(name='proto')
def proto_cli():
    pass


@proto_cli.command()
#this function takes in the original log and extracts needed variables
def condense():
        output = []
        with open ('data/protobowl-042818.log') as f:
            for line in f:
                data = json.loads(line)
                #there are 2k items with question_id instead of an actual id. Skip them.
                if data['object']['qid'] == 'question_id':
                    continue
                #time_remaining is ACTUAL total time in the original log
                guess_ratio = round(data['object']['time_elapsed'] /data['object']['time_remaining'],4)
                #pull out relevant variables
                output.append({'qid':data['object']['qid'], 
                               'user_id':data['object']['user']['id'], 
                               'guess_ratio':guess_ratio, 
                               'correct_answer_bool':data['object']['ruling'], 
                               'date':data['date']}
                             )
                line = f.readline()

        with open('data/protobowl-042818-condensed.log', 'w') as f:
            json.dump({'data':output}, f)

    #make_hashes runs after the condensed log is created.  It creates a hash that stores statistics
    #for each user and for each question. Since the log is unstructured, generating both hashes takes ~13 hours
    def make_hashes():        
        with open('data/protobowl-042818-condensed.log', 'r') as f:
            log = json.load(f)

        unique_qid = set([item['qid'] for item in log['data'] if item['qid'] !='question_id'])
        print("Example data being summarized:\n", log['data'][0])

        qid_hash = {}
        skipped_counter = 0

        #120k unique question IDs in log data. 
        for qid in tqdm_notebook(unique_qid, total = len(unique_qid)):
            qid_dict = {}
            #find matching ids in all data.  Drop questions that don't have a T/F answer value.
            grouped_data = [(data['user_id'], data['correct_answer_bool'], data['guess_ratio']) for data in log['data'] 
                            if data['qid'] == qid and data['correct_answer_bool']!='prompt']

            #some questions may only have "prompt" as response
            if len(grouped_data) == 0:
                skipped_counter +=1
                continue

            else:
                #technically unneeded code for variable transparency
                users_per_question = [tupple[0] for tupple in grouped_data]
                accuracy_per_question = [tupple[1] for tupple in grouped_data]
                length_per_question = [tupple[2] for tupple in grouped_data] 

                #calculate ratio of answers as # Trues / # Trues + # False
                accuracy_ratio = round(sum(accuracy_per_question)/len(accuracy_per_question), 4)
                length_ratio = round(sum(length_per_question)/len(length_per_question), 4)

                #store in dictionary, and then store that in hash
                qid_dict['users_per_question'] = users_per_question
                qid_dict['accuracy_per_question'] = accuracy_per_question
                qid_dict['length_per_question'] = length_per_question
                qid_dict['overall_length_per_question'] = length_ratio
                qid_dict['overall_accuracy_per_question'] = accuracy_ratio
                qid_hash[qid] =  qid_dict

        print(f"Finished.  Skipped {skipped_counter} questions that didn't have responses.")
        with open('protobowl_byquestion_hash.json', 'w') as f:
            json.dump({'data':qid_hash}, f)
            

#after the hashes are created, we can update the qanta data to have question_stats built in.           
def update_qanta_splits():
    
    #go through each qanta split and append relevant statistics.
    #drop instances where no proto_id exists in the qanta data.

    for split in ['train', 'dev', 'test']:

        with open(f'data/qanta.{split}.2018.04.18.json', 'r') as f:
            qanta_data = json.load(f)

        with open('protobowl_byquestion_hash.json', 'r') as f:
            question_stats = json.load(f)['data']

        qids = [keys for keys in question_stats]

        q_counter = 0
        updated_qanta_questions = []
        for q in tqdm_notebook(qanta_data['questions'], total=len(qanta_data['questions'])):
            if q['proto_id'] in qids:
                selected_q = question_stats[q['proto_id']]
                q['question_stats'] = {'users_per_question':selected_q['users_per_question'], 
                                       'length_per_question':selected_q['length_per_question'],  
                                       'overall_length_per_question':selected_q['overall_length_per_question'], 
                                       'accuracy_per_question':selected_q['accuracy_per_question'], 
                                       'overall_accuracy_per_question':selected_q['overall_accuracy_per_question']}
                updated_qanta_questions.append(q)    
                q_counter+=1

        print(f"Updated data has {q_counter} entrees.  Shrunk from total qanta split of {len(qanta_data['questions'])}")
        with open(f'data/expanded.qanta.{split}.2018.04.18.json', 'w') as f:
            json.dump({'questions':updated_qanta_questions}, f)
