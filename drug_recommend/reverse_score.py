import os
import pandas as pd
import numpy as np
import sys


class recommened():
    def __init__(self, drug_path):
        self.drug_path = drug_path

    '''
        This function takes q, a list of gene names, and r1, a panda data
        frame as the input, and output the enrichment score vector
    '''
    def computees(self, q, r1):
        
        if len(q) == 0:
            ks = 0
        elif len(q) == 1:
            ks = r1.loc[q,:]
            ks.index = [0]
            ks = ks.T
        #print(ks)
        else:
            n = r1.shape[0]      # n=12328
            sub = r1.loc[q,:]
            J = sub.rank()
            a_vect = J/len(q)-sub/n
            b_vect = (sub-1)/n-(J-1)/len(q) 
            #b_vect = (sub)/n-(J-1)/len(q)
            a = a_vect.max()
            b = b_vect.max()
            ks = []
            for i in range(len(a)):
                if a[i] > b[i]:
                    ks.append(a[i])
                else:
                    ks.append(-b[i])
        #print(ks)
        return ks


    def ranklist(self, DT):
    # This function takes a panda data frame of gene names and expressions
    # as an input, and output a data frame of gene names and ranks
        ranks = DT.rank(ascending=False, method="first")
        return ranks


    # This file consists of useful functions that are related to cmap. 
    # Reference: https://github.com/kekegg/DLEPS
    def computecs(self, qup, qdown, expression):
        '''
        This function takes qup & qdown, which are lists of gene
        names, and  expression, a panda data frame of the expressions
        of genes as input, and output the connectivity score vector
        '''
        r1 = self.ranklist(expression)
        if qup and qdown:
            esup = self.computees(qup, r1)
            # print('esup:', esup)
            esdown = self.computees(qdown, r1)
            # print('esdown:', esdown)
            w = []
            for i in range(len(esup)):
                if esup[i]*esdown[i] <= 0:
                    w.append(esup[i]-esdown[i])
                else:
                    w.append(0)
            return pd.DataFrame(w, expression.columns)
        elif qup and qdown==None:
            esup = self.computees(qup, r1)
            return pd.DataFrame(esup, expression.columns)
        elif qup == None and qdown:
            esdown = self.computees(qdown, r1)
            return pd.DataFrame(esdown, expression.columns)
        else:
            return None


    def drug_recommened(self, delta_gene_express, up_gene, down_gene, reverse_score):

        drug_fc = pd.read_csv(os.path.join(self.drug_path, delta_gene_express), index_col=0)

        up_gene_list = np.loadtxt(os.path.join(self.drug_path, up_gene), dtype=str)
        down_gene_list = np.loadtxt(os.path.join(self.drug_path, down_gene), dtype=str)

        gene_list = drug_fc.columns.tolist()
        up_gene_list_ = up_gene_list.tolist()
        print('up_gene_list_:', len(up_gene_list_))
        down_gene_list_ = down_gene_list.tolist()
        print('down_gene_list_:', len(down_gene_list_))

        up_gene_list = list(set(gene_list) & set(up_gene_list_))
        down_gene_list = list(set(gene_list) & set(down_gene_list_))
        print('Genes included in L1000')
        print('up_gene_list:', len(up_gene_list))
        print('down_gene_list:', len(down_gene_list))

        if not up_gene_list:
            up_gene_list=None
        if not down_gene_list:
            down_gene_list=None

        fc_es_array = self.computecs(up_gene_list, down_gene_list, drug_fc.T)[0].values

        es_df = pd.DataFrame(fc_es_array,index=drug_fc.index,columns=['fold-change es'])

        recommened_drug_sort = es_df.sort_values(by='fold-change es', ascending=True)

        recommened_drug_sort.to_csv(reverse_score)
        # print(recommened_drug_sort)

        recommened_drug_sort_min = es_df.sort_values(by='fold-change es', ascending=True).head(20)
        # print(recommened_drug_sort_min)

        recommened_drug_sort_max = es_df.sort_values(by='fold-change es', ascending=False).head(20)
        # print(recommened_drug_sort_max)

        return recommened_drug_sort_max, recommened_drug_sort_min


recommen = recommened('./')
# disease = 'melanoma'
# drug_sort_max, drug_sort_min = recommen.drug_recommened(delta_gene_express=f'./topscience/d006/{disease}_cancer/merged_mean_result.csv',
#                                                         up_gene=f'embl-ebi_maker_gene/{disease}_cancer_deg/{disease}_disease_up_genes.txt',
#                                                         down_gene=f'embl-ebi_maker_gene/{disease}_cancer_deg/{disease}_disease_down_genes.txt',
#                                                         reverse_score=f'./topscience/d006/{disease}_cancer/merged_mean_result_reverse_score-embl.csv')

drug_sort_max, drug_sort_min = recommen.drug_recommened(delta_gene_express=sys.argv[1],
                                                        up_gene=sys.argv[2],
                                                        down_gene=sys.argv[3],
                                                        reverse_score=sys.argv[4])

print(drug_sort_max)
print(drug_sort_min)
