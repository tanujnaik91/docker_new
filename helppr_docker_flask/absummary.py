from transformers import T5Tokenizer, T5Model,pipeline
from paragrapher import break_into_sentences, capitalise_first_sentence
import time

MODEL_TYPE = "t5-base"

TOKENIZER_MAX_SEQ_LENGTH = 512

def custom_summarize(text, min_summary_length=5,max_summary_length=200):
    summarizer = pipeline("summarization", model=MODEL_TYPE,tokenizer=MODEL_TYPE)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_TYPE)
    encoded_seq = tokenizer(text)
    enc_input_ids = encoded_seq["input_ids"]
    done = False

    print("Generating Summaries of max length: ",max_summary_length)

    if max_summary_length > TOKENIZER_MAX_SEQ_LENGTH:
        print("'max_summary_length' cannot be more than 512, reducing the value to 512")
        max_summary_length = TOKENIZER_MAX_SEQ_LENGTH

    if len(enc_input_ids) > TOKENIZER_MAX_SEQ_LENGTH:
        #if you enter here, this means the tokens are more than 512.
        #In this case let us figure out a way to intelligently break out the text so that it falls under 512 tokens

        #this variable is created to handle information rentention ratio
        bins = len(enc_input_ids) * 0.05

        #this variable is created to learn how many times the 'while' loop got executed
        counter = 0

        candidate = text
        old_target_text, old_pending_text, target_text, pending_text = '','','',''
        counter = 0
        start_time = time.time()
        while not done:
                counter = counter + 1
                old_target_text = target_text
                old_pending_text = pending_text
                target_text, pending_text, custom_size = split_text(candidate,TOKENIZER_MAX_SEQ_LENGTH, tokenizer)
                if old_pending_text == pending_text:
                    print("\n","Pending text is same as before")
                if old_target_text == target_text:
                    print("Target text is same as before","\n")
                part_sum = summarizer(target_text, max_length = custom_size, min_length = int(custom_size*information_retention_ratio(bins, counter)))
                if pending_text is None:
                    print(target_text)
                    done = True
                    summary = part_sum
                    print(counter)
                else:
                    candidate = part_sum[0]['summary_text'] + pending_text
                    counter = counter + 1
        print("While loop took time: ", time.time()-start_time)
    else:
         summary = summarizer(text, min_length = min_summary_length, max_length=max_summary_length)

    return capitalise_first_sentence(summary[0]['summary_text'])


def split_text(text, max_len, tokenizer):
    text_list = break_into_sentences(text)

    #this codesnippet is important for all 1+ trials to avoid infinite loop
    curr_len = len(tokenizer(text)['input_ids'])
    if curr_len < max_len:
        return text, None, curr_len

    #this needs to be revisited to intellignetly design the position logic for large indexes
    #Currently we re doing it sequential traversal from last index
    position = len(text_list) - 1

    #this code will always run for first call
    done = False
    target_text = None
    pending_text = None
    while not done:
        part_text = ' '.join(text_list[:position])
        curr_len = len(tokenizer(part_text)['input_ids'])
        if curr_len > max_len and position > -1:
            position = position - 1

        elif position < 0:
            #this needs to be more intellignetly handled, beacuse this means
            #that sentences group or sentence at position 0
            #is very BIG!
            target_text = text_list[0]
            pending_text = ' '.join(text_list[1:])


        else:
            target_text = part_text
            try:
                pending_text = ' '.join(text_list[position:])
            except:
                pending_text = None

        if target_text is not None:
            done= True

    return target_text, pending_text, curr_len

def information_retention_ratio(bins, counter,eqn ='linear', base_val=0.05,max_val=1):
    if eqn == 'linear':
        slope = (max_val - base_val)/(bins-1)  #note x1 = 1 not zero
        ratio = slope*(counter-1) + base_val
        if ratio > 1:
           ratio = 0.9
        return ratio
