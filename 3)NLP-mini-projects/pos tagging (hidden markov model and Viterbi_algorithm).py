# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 13:31:49 2020

@author: Mitran
"""

import numpy as np
import pandas as pd
#Emission Probability
def emissionprob(word,tag,dataset):
    # probabilit that a given word is the particular part of speech
    tag_list=[tup for tup in dataset if tup[1]==tag]
    word_gn_tag =[tup[0] for tup in tag_list if tup[0]==word]
    
    return(len(word_gn_tag),len(tag_list)) #word_gn_tag numerator        tag_list denominator
                                            # no of time a particular words (..say john) appeared as particular(pos) (..say Noun)  /total no appearance fo pos (..say Noun)

#Transition Probability
def transitionprob(t2,t1,dataset):
    # probabilit that a given particular (pos)tag will follow another or same (pos)tag   EG   prob that modal verb will come after noun
    tags= [pair[1] for pair in dataset]
    succesor_tag=[tag for tag in tags if tag==t1 ]
    
    count_t2_t1 = 0
    for i in range(len(tags)-1):
        #t1=noun  t2 =verb check number of times ((noun,  verb )) appears
        if(tags[i]==t1 and tags[i+1]==t2):
            count_t2_t1+=1
    return count_t2_t1 ,len(succesor_tag)  #count_t2_t1 numerator         succesor_tag denominator
                                            # no of time a co-occurence tag (..say NOUN VERB)  /total no appearance the tag is succesor for all the combination (..say 'Noun VERB' 'NOUN NOUN' 'NOUN MODALVERB' 'NOUN CONJUCTION')
    
def hidden_markov_model(ds):
    tags={tag for ln in ds for word,tag in ln}
    vocab={word for ln in ds for word,tag in ln}
    
    train_tag_word=[tup for sent in ds for tup in sent]
    ta=list(tags)
    vo=list(vocab)
    tpmat=np.zeros((len(tags),len(tags)),dtype='float32')
    epmat=np.zeros((len(vo),len(tags)),dtype='float32')
    
    for i,t1 in enumerate(ta):
      for j,t2 in enumerate(ta):
        num,den=transitionprob(t2,t1,train_tag_word)        
        tpmat[i,j]=num/den
        """
        if t1=='.':
            break
            """


    for i,w in enumerate(vo):
      for j,t in enumerate(ta):        
        num,den=emissionprob(w,t,train_tag_word)  
        epmat[i,j]=num/den
        
        
    tran_prob_df=pd.DataFrame(tpmat,columns=ta,index=ta)
    
    
    
    emi_prob_df=pd.DataFrame(epmat,columns=ta,index=vo)
    
    
    return train_tag_word,emi_prob_df,tran_prob_df
    
    
def merge(sent,s):
    ds=[]
    for se,sd in zip(sent,s):
      lm=[]
      for w,pos in zip(se.split(),sd.split()):
          lm.append([w,pos])
      ds.append(lm)
    return ds
 
def Viterbi_algorithm(words, train_bag,tags_df,emis_prob):
    state = [] # emission* transition
    tags = list(set([pair[1] for pair in train_bag]))
    
    for i,word in enumerate(words):
       # print(i,word)
        #initialise list of probability column for a given observation
        p = [] 
        for t in tags:
            """Dynamic Programming (DP) is an algorithmic technique for solving an optimization problem by breaking it down into simpler subproblems and utilizing the fact that the optimal solution to the overall problem depends upon the optimal solution to its subproblems."""
            if i == 0:
                transition_p = tags_df.loc['.', t]
            else:
                transition_p = tags_df.loc[state[-1], t]
                
            emission_p=emis_prob.loc[word,t]
            print(word,t,'\t',transition_p,t)
            state_probability = emission_p * transition_p    
            p.append(state_probability)
            
            
        
         # getting state for which probability is maximum
        state_max = tags[p.index(max(p))] 
        print(state_max,max(p))
        state.append(state_max)
        
    return list(zip(words, state))
    
def Viterbi_algorithm_with_rule(words, train_bag,tags_df,emis_prob,rule):
    state = [] # emission* transition
    tags = list(set([pair[1] for pair in train_bag]))
    
    for i,word in enumerate(words):
       # print(i,word)
        #initialise list of probability column for a given observation
        p = [] 
        for t in tags:
            """Dynamic Programming (DP) is an algorithmic technique for solving an optimization problem by breaking it down into simpler subproblems and utilizing the fact that the optimal solution to the overall problem depends upon the optimal solution to its subproblems."""
            if i == 0:
                transition_p = tags_df.loc['.', t]
            else:
                transition_p = tags_df.loc[state[-1], t]
                
            emission_p=emis_prob.loc[word,t]
            print(word,t,'\t',transition_p,t)
            state_probability = emission_p * transition_p    
            p.append(state_probability)            
            
            
           
        
        state_max = rule.tag([word])[0][1]
        if(max(p)==0):
            state_max = rule.tag([word])[0][1] # assign based on rule based tagger
        else:
            if state_max != 'X':
               state_max = tags[p.index(max(p))]            
            
        state.append(state_max)
        
    return list(zip(words, state))  
    

sent=['mary jane can see will .','spot will see mary .','will jane spot mary .','mary will pat spot .']
s=['N N M V N .','N M V N .','M N V N .','N M V N .']
   
processed_Data,emission_prob,transition_prob=hidden_markov_model(merge(sent,s))

"""The Viterbi algorithm is a dynamic programming algorithm for finding the most likely sequence of hidden states—called the Viterbi path—that results in a sequence of observed events, especially in the context of Markov information sources and hidden Markov models (HMM)."""
test_sent=['jane','can','spot','spot']
# viterbi fro finding optimal sol by forward and back pro
new=Viterbi_algorithm(test_sent,processed_Data,transition_prob,emission_prob)


    

#To improve the performance,we specify a rule base tagger for unknown words 
# specify patterns for tagging
patterns = [
    (r'.*ing$', 'V'),              # gerund
    (r'.*ed$', 'V'),               # past tense 
    (r'.*es$', 'V'),               # verb    
    (r'.*\'s$', 'N'),              # possessive nouns
    (r'.*s$', 'N'),               # plural nouns    
    (r'.*', 'N')                   # nouns
]
"""(r'\*T?\*?-[0-9]+$', 'X'),        # X
    (r'^-?[0-9]+(.[0-9]+)?$', 'NUM'), # cardinal numbers"""

import nltk
#new1=Viterbi_algorithm_with_rule(test_sent,processed_Data,transition_prob,emission_prob,nltk.RegexpTagger(patterns))
    
    

    


# nltk_data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))   more examples data




        
  




        
        
