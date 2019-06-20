import click
import json
from tqdm import tqdm_notebook

@click.group(name='proto')
def proto_cli():
    pass


@proto_cli.command()
def process():
    output = []
    with open ('protobowl-042818.log') as f:
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

    with open('protobowl-042818-condensed.log', 'w') as f:
        json.dump({'data':output}, f)
