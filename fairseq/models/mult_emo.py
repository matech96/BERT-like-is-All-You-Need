# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
RoBERTa: A Robustly Optimized BERT Pretraining Approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import logging

from fairseq import utils
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
    TransformerMultiEncoder
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
#from fairseq.models.roberta import RobertaModel
from fairseq import checkpoint_utils
import pdb
from fairseq.models.roberta import RobertaModel


logger = logging.getLogger(__name__)
#from .hub_interface import RobertaHubInterface


@register_model('robertaEMO')
class RobertaEMOModel(FairseqLanguageModel):


    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

       

        ############################## Adding the pretrained SSL models to extract features###############

        if self.args.a_only or self.args.all_in:
           
            # self.roberta_vqwav2vec = RobertaModel.from_pretrained('/hpc/gsir059/phd1st/trained_ssl/wav2vec/vq-wav2vec-Kmeans-Roberta', checkpoint_file='bert_kmeans.pt')
            self.roberta_vqwav2vec = RobertaModel.from_pretrained('pretrained_ssl/roberta.large/', checkpoint_file='bert_kmeans.pt')

            # for param in  self.roberta_vqwav2vec.parameters():
            #     param.requires_grad = False


        if self.args.t_only or self.args.all_in:
            roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')

        
            ########################### Freezing pretrained SSL paramtere###################################
            self.model_text2vec=roberta
            # for param in self.model_text2vec.parameters():
            #     param.requires_grad = False




       

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers in plain trans')
        parser.add_argument('--encoder-layers-cross', type=int, metavar='L',
                            help='num encoder layers in cross modal trans')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-dim-concat', type=int, metavar='H',
                            help='encoder embedding dimension concatenated')
        parser.add_argument('--encoder-embed-dim-a', type=int, metavar='H',
                            help='encoder embedding dimension cross modal audio')
        parser.add_argument('--encoder-embed-dim-v', type=int, metavar='H',
                            help='encoder embedding dimension cross modal audio')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions-t', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')

        parser.add_argument('--max-positions-v', type=int,
                            help='number of positional embeddings to learn in video stream')
        parser.add_argument('--max-positions-a', type=int,
                            help='number of positional embeddings to learn in audio stream')


        parser.add_argument('--t-only', action='store_true', default=False,
                            help='do you need only text')

        parser.add_argument('--v-only', action='store_true', default=False,
                                    help='do you need only video')

        parser.add_argument('--a-only', action='store_true', default=False,
                                    help='do you need only audio')


        parser.add_argument('--all-in', action='store_true', default=False,
                                    help='do you need all the embeddings')

        parser.add_argument('--stack-up', action='store_true', default=False,
                                    help='do you need to add an architecture on top of SSL layers')


    
    def set_num_updates(self, num_updates):

        self.curr_step = num_updates


    
    @classmethod
    def build_model(cls, args, task):


        """Build a new model instance."""
        # make sure all arguments are present
        #base_architecture(args) #Supplying the relevence arguments to create the baseline transformer architecture

       

        encoder = RobertaEMOEncoder(args, task.source_dictionary)
     
        return cls(args, encoder)

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, classification_head_name=None, **kwargs):
        

        data_dict={} #Create a dictionary with features
        data_dict['raw_data']=src_tokens

        if self.args.t_only or self.args.all_in:
            tokens_text=src_tokens['text']
        

            #Text SSL feature extraction  # [2, 100, 1024] B X T X D
            roberta_feature=self.model_text2vec.extract_features(tokens_text)
            data_dict['Text']=roberta_feature
        
                

        if self.args.a_only or self.args.all_in:
            tokens_audio=src_tokens['audio']


            roberta_vqwav2vec_feature=self.roberta_vqwav2vec.extract_features(tokens_audio)
    
            data_dict['Audio']=roberta_vqwav2vec_feature
    

            #Audio SSL feature extraction [2, 512, 310] B X D X T

        # if self.args.v_only or self.args.all_in:
        #     data_dict['Video']=src_tokens['embedded_video']

    
        if classification_head_name is not None:
            features_only = True

        

        #This will output the main models whole features as well as token features
        x, extra = self.decoder(data_dict,features_only, return_all_hiddens, **kwargs) #here the decoder means the encoder (to have the interface fixed)
        if True: # self.args.v_only or self.args.all_in:
            extra['j_video'] = src_tokens['embedded_video']

        
        if classification_head_name is not None:
        
            x = self.classification_heads[classification_head_name](extra) #Here we send the extracs ("First token")
        
        
       
        return x, extra

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""

        
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                print(
                    'WARNING: re-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

    
        self.classification_heads[name] = RobertaEMOClassificationHead(
            self.args.encoder_embed_dim_concat,#self.args.encoder_embed_dim,
            #inner_dim or self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim_concat,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
            self.args
        )

    @property
    def supported_targets(self):
        return {'self'}


    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

     

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            #print(k)

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            #inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)
            #Because we do not have a dense layer
            inner_dim =state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(1)

    

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    # or inner_dim != self.classification_heads[head_name].dense.out_features
                    or inner_dim != self.classification_heads[head_name].out_proj.in_features #because we do not have a dense layer
                    
                ):
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)

      
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    print('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v








def upgrade_state_dict_named(self, state_dict, name):
    super().upgrade_state_dict_named(state_dict, name)

    prefix = name + '.' if name != '' else ''
    current_head_names = [] if not hasattr(self, 'classification_heads') else \
        self.classification_heads.keys()

    # Handle new classification heads present in the state dict.
    keys_to_delete = []
    for k in state_dict.keys():
        if not k.startswith(prefix + 'classification_heads.'):
            continue

        head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
        num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
        #inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)
        inner_dim =state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(1)

        if getattr(self.args, 'load_checkpoint_heads', False):
            if head_name not in current_head_names:
                self.register_classification_head(head_name, num_classes, inner_dim)
        else:
            if head_name not in current_head_names:
                logger.warning(
                    'deleting classification head ({}) from checkpoint '
                    'not present in current model: {}'.format(head_name, k)
                )
                keys_to_delete.append(k)
            elif (
                num_classes != self.classification_heads[head_name].out_proj.out_features
                or inner_dim != self.classification_heads[head_name].dense.out_features
            ):
                logger.warning(
                    'deleting classification head ({}) from checkpoint '
                    'with different dimensions than current model: {}'.format(head_name, k)
                )
                keys_to_delete.append(k)
    for k in keys_to_delete:
        del state_dict[k]

    # Copy any newly-added classification heads into the state dict
    # with their current weights.
    if hasattr(self, 'classification_heads'):
        cur_state = self.classification_heads.state_dict()
        for k, v in cur_state.items():
            if prefix + 'classification_heads.' + k not in state_dict:
                logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                state_dict[prefix + 'classification_heads.' + k] = v






class RobertaEMOClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout,args):
        super().__init__()

        #inner_dim needs to change

        #self.dense = nn.Linear(input_dim, inner_dim)
        #self.activation_fn = utils.get_activation_fn(activation_fn) #Activation function is Tanh
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)


        self.args=args

    def forward(self, features, **kwargs):


        T=features['j_text']

        A=features['j_aud']

        V=features['j_video']
      
        Final=torch.cat((T,A,V),dim=1)
        

    
        x =Final# features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        # print(x.shape, (self.out_proj.in_features, self.out_proj.out_features))
        x = self.out_proj(x)
       
        return x


class RobertaEMOEncoder(FairseqDecoder):
    """RoBERTa encoder.

    Implements the :class:`~fairseq.models.FairseqDecoder` interface required
    by :class:`~fairseq.models.FairseqLanguageModel`.
    """

    def __init__(self, args, dictionary):
  
        super().__init__(dictionary)

        self.args = args


        self.sentence_encoder = TransformerMultiEncoder(     #This encodes all the 
            padding_idx=1,#dictionary.pad(), #Check!
            #vocab_size=0,#len(dictionary),#Check!
            #num_encoder_layers=args.encoder_layers,
            num_encoder_layers_cross=args.encoder_layers_cross,
            #embedding_dim=args.encoder_embed_dim,
            embedding_dim_text=args.encoder_embed_dim_t,
            embedding_dim_audio=args.encoder_embed_dim_a,
            #embedding_dim_video=args.encoder_embed_dim_v,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            #max_seq_len_text=args.max_positions_t,#This is the text
            #max_seq_len_audio=args.max_positions_a,
            #max_seq_len_video=args.max_positions_v,
            #num_segments=0,
            #encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            is_only_text=args.t_only,
            is_only_audio=args.a_only,
            #is_only_video=args.v_only,
            is_all_in=args.all_in,
            is_stack_up=args.stack_up

        )

          


    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the  LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states.
        """
     
        x, extra = self.extract_features(src_tokens, return_all_hiddens) #Get the output after cross attentions and self attention mechanisms

       
        #Next track the loss function
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):  #This 
   
        
        inner_states, seq_rep = self.sentence_encoder(  #use the foewardfunction in TransformerMultiEncoder object
            src_tokens, last_state_only=not return_all_hiddens,
        )

        return inner_states,seq_rep

    def max_positions(self):
        """Maximum length supported by the model."""
        return None #self.args.max_positions
        #return sys.maxsize



@register_model_architecture('robertaEMO', 'robertEMO_large')
def robertaEMO_large_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_layers_cross = getattr(args, 'encoder_layers_cross', 1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024) 
    # args.encoder_embed_dim_concat = getattr(args, 'encoder_embed_dim_concat',1792 )  #2048(1024 + 512 + 256) # audio and text
    args.encoder_embed_dim_concat = getattr(args, 'encoder_embed_dim_concat',2304 )  #2048(1024 + 512 + 256) # all
    #args.encoder_embed_dim_concat = getattr(args, 'encoder_embed_dim_concat',1280 )  #2048(1024 + 512 + 256) #audio only
    args.encoder_embed_dim_t = getattr(args, 'encoder_embed_dim_t', 1024) 
    args.encoder_embed_dim_a = getattr(args, 'encoder_embed_dim_a', 768) 
    args.encoder_embed_dim_v = getattr(args, 'encoder_embed_dim_v', 512) 
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024) #previously this was 1024
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)


