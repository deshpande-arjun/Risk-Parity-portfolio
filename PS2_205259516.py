# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 19:55:23 2019

@author: arjun
"""

import numpy as np
import pandas as pd
import os

os.getcwd()
path = "E:/3_Quarter/Quantitative Asset Management/PS 2"
os.chdir(path)
all_bonds = pd.read_csv('all_bonds_since1926.csv')

input_variables_bonds = np.array(['KYCRSPID', 'MCALDT', 'TMRETNUA', 'TMTOTOUT'])
all_bonds = all_bonds[input_variables_bonds]
#change dtype of MCALDT to date:
all_bonds.MCALDT = pd.to_datetime(pd.Series(all_bonds.MCALDT),format="%Y-%m-%d")
#change CRSP ID into string:
all_bonds['KYCRSPID']= all_bonds.KYCRSPID.astype(str)
all_bonds.dtypes


all_bonds.head()


def PS2_Q1(all_bonds):
    all_bonds.TMRETNUA.value_counts()
    #when price is missing of this month or previous month, the Return is set to -99.00
    #Thus we should remove those observations
    all_bonds = all_bonds[(all_bonds['TMRETNUA']!=-99) ]

    #check if there are any missing values in Returns:
    sum(all_bonds.TMRETNUA.isnull())
    #check if there are any missing values in total amount of bonds issued in Million dollars:
    sum(all_bonds.TMTOTOUT.isnull())
    #remove those observations:
    all_bonds = all_bonds[-all_bonds.TMTOTOUT.isnull()]
    
    #create mkt cap variable:
    all_bonds['Bond MV'] = all_bonds.TMTOTOUT
    #lag the variable:
    all_bonds = all_bonds.set_index(['MCALDT','KYCRSPID'])
    all_bonds['Bond_lag_MV'] = all_bonds['Bond MV'].groupby(level = "KYCRSPID").shift(1)
    
    all_bonds = all_bonds.reset_index()
    all_bonds = all_bonds.set_index(['MCALDT'])

    #remove those observations without mkt value:
#    all_bonds = all_bonds[-all_bonds.Bond_lag_MV.isnull()]

    #create equal weighted returns:
    all_bonds['Bond_Ew_Ret'] = all_bonds.TMRETNUA.groupby(level = 'MCALDT').mean()

    #create value weighted returns:
    all_bonds['Bond_Vw_Ret'] = all_bonds['TMRETNUA']*all_bonds['Bond_lag_MV']
    all_bonds['Bond_Vw_Ret'] = all_bonds['Bond_Vw_Ret'].groupby(level = "MCALDT").sum()/all_bonds['Bond_lag_MV'].groupby(level = "MCALDT").sum()

    #sum up mkt cap for each period: 
    all_bonds['Bond_lag_MV'] = all_bonds.Bond_lag_MV.groupby(level = "MCALDT").sum()
    
    #reset index:
    all_bonds = all_bonds.reset_index()
    
    #change date into a YYYYMMDD format:
    all_bonds.MCALDT = pd.to_datetime(pd.Series(all_bonds.MCALDT),format="%Y%m%d")
    
    #transform date into year and month columns:
    all_bonds['Year'] = pd.DatetimeIndex(all_bonds.MCALDT).year
    all_bonds['Month'] = pd.DatetimeIndex(all_bonds.MCALDT).month
    
    #remove duplicates & sort it according to year & date:
    all_bonds = all_bonds.drop_duplicates(subset = ['Year', 'Month'])
    all_bonds = all_bonds.sort_values(by = ['Year', 'Month'])
    
###    all_bonds = all_bonds.fillna(0)
  
    #keep only useful variables:    
    output_variables = np.array(['Year', 'Month', 'Bond_lag_MV', 'Bond_Ew_Ret', 'Bond_Vw_Ret'])
    all_bonds = all_bonds[output_variables]
    
    return(all_bonds)
    

Q1_ans = PS2_Q1(all_bonds)    


# =============================================================================
#Question 2:

##INPUTS:

all_stocks = pd.read_csv("all_stocks_since1926.csv")
input_variables_stocks = np.array(['PERMNO','date','SHRCD','EXCHCD','RET','DLRET','PRC','SHROUT'])
all_stocks = all_stocks[input_variables_stocks]

def PS1_Q1(all_stocks):
    #check exchange codes of stocks & keep only NYSE & NASDAQ i.e. 1,2,3
    all_stocks['EXCHCD'].value_counts()
    all_stocks = all_stocks[(all_stocks['EXCHCD']==3) | (all_stocks['EXCHCD']==2) | (all_stocks['EXCHCD']==1)]
    #keep data for share codes 10 & 11:
    all_stocks = all_stocks[(all_stocks['SHRCD']==10) | (all_stocks['SHRCD']==11)]
    #remove observations when RET is A,B,C which means no valid previous price:
    all_stocks = all_stocks[(all_stocks.RET !='C') & (all_stocks.RET !='B') & (all_stocks.RET !='A') ]
    #removing entries where delisting returns are "A" or "S" or "T" or "P"
    all_stocks =  all_stocks[(all_stocks['DLRET'] != 'A') & (all_stocks['DLRET'] != 'S') & (all_stocks.DLRET!='P') & (all_stocks.DLRET!='T')]
    #remove if both RET & DLRET is NaN:
    all_stocks = all_stocks[-(all_stocks.RET.isnull()) | -(all_stocks.DLRET.isnull()) ]
    all_stocks['RET'] = all_stocks['RET'].astype(float)
    all_stocks['DLRET'] = all_stocks['DLRET'].astype(float)
    
    #convert outstanding to $Millions from $thousands
    all_stocks['Stock_lag_MV'] = (1/1000)*np.abs(all_stocks['PRC'])*all_stocks['SHROUT']
    all_stocks = all_stocks.set_index(['date','PERMNO'])
    all_stocks['Stock_lag_MV'] = all_stocks['Stock_lag_MV'].groupby(level = "PERMNO").shift(1)
    all_stocks['RET_combined'] = (1+all_stocks['RET'])*(1+all_stocks['DLRET']) - 1
    all_stocks.loc[pd.isnull(all_stocks['RET']),'RET_combined'] = all_stocks.loc[pd.isnull(all_stocks['RET']),'DLRET']
    all_stocks.loc[pd.isnull(all_stocks['DLRET']),'RET_combined'] = all_stocks.loc[pd.isnull(all_stocks['DLRET']),'RET']

    all_stocks = all_stocks.reset_index()
    all_stocks = all_stocks.set_index(['date'])
 
#    all_stocks = all_stocks[~pd.isnull(all_stocks['RET_combined'])]
#    all_stocks = all_stocks[~pd.isnull(all_stocks['Stock_lag_MV'])]

    all_stocks['Stock_Ew_Ret'] = all_stocks['RET_combined'].groupby(level = "date").mean()
    all_stocks['Stock_Vw_Ret'] = all_stocks['RET_combined']*all_stocks['Stock_lag_MV']
    all_stocks['Stock_Vw_Ret'] = all_stocks['Stock_Vw_Ret'].groupby(level = "date").sum()/all_stocks['Stock_lag_MV'].groupby(level = "date").sum()
    all_stocks['Stock_lag_MV'] = all_stocks['Stock_lag_MV'].groupby(level = "date").sum()
    all_stocks = all_stocks.reset_index()

    all_stocks["date"] = pd.to_datetime(pd.Series(all_stocks["date"]), format = "%Y%m%d" )
    all_stocks['Year'] = pd.DatetimeIndex(all_stocks.date).year 
    all_stocks['Month'] = pd.DatetimeIndex(all_stocks.date).month

###    all_stocks_mod = all_stocks.fillna(0)
    
    all_stocks_mod = all_stocks
    all_stocks_mod = all_stocks_mod.drop_duplicates(subset = ['Year','Month'])
    all_stocks_mod = all_stocks_mod.sort_values(by = ['Year','Month'])

    output_variables_stocks = np.array(['Year','Month','Stock_lag_MV','Stock_Ew_Ret','Stock_Vw_Ret'])
    all_stocks_mod = all_stocks_mod[output_variables_stocks]

    return(all_stocks_mod)

Monthly_CRSP_Riskless = pd.read_csv('Monthly_CRSP_Riskless.csv')
#change to proper date format:
Monthly_CRSP_Riskless['caldt'] = pd.to_datetime(pd.Series(Monthly_CRSP_Riskless.caldt), format= "%Y%m%d")

Monthly_CRSP_Stocks = PS1_Q1(all_stocks)
Monthly_CRSP_Bonds = PS2_Q1(all_bonds)

#OUTPUT:

def PS2_Q2(Monthly_CRSP_Stocks,Monthly_CRSP_Bonds,Monthly_CRSP_Riskless):

    #transform date into year and month columns:
    Monthly_CRSP_Riskless['Year'] = pd.DatetimeIndex(Monthly_CRSP_Riskless.caldt).year
    Monthly_CRSP_Riskless['Month'] = pd.DatetimeIndex(Monthly_CRSP_Riskless.caldt).month
    
    #sort it according to year & date:
    Monthly_CRSP_Riskless = Monthly_CRSP_Riskless.sort_values(by = ['Year', 'Month'])
    
    #merge data, first stocks with bonds, then join riskless to this dataframe:    
    add1 = pd.merge(Monthly_CRSP_Stocks,Monthly_CRSP_Bonds,how='outer',on=['Year','Month'])
    add2 = pd.merge(add1, Monthly_CRSP_Riskless ,how='outer',on=['Year','Month'])
    
    #Since we are considering monthly returns, take 30 day Treasury:
    add2['Stock_Excess_Vw_Ret'] = add2.Stock_Vw_Ret - add2.t30ret
    add2['Bond_Excess_Vw_Ret'] = add2.Bond_Vw_Ret - add2.t30ret

    #keep only the required variables:    
    output_variables_merged = np.array(['Year','Month','Stock_lag_MV','Stock_Excess_Vw_Ret','Bond_lag_MV','Bond_Excess_Vw_Ret'])
    add2 = add2[output_variables_merged]
    add2.Bond_lag_MV = add2.Bond_lag_MV.astype(int)
#    add2.Bond_lag_MV = pd.to_numeric(add2.Bond_lag_MV)
    add2.dtypes
    
    return(add2)    
    

Q2_ans = PS2_Q2(Monthly_CRSP_Stocks,Monthly_CRSP_Bonds,Monthly_CRSP_Riskless)

Q2_ans.mean()

# =============================================================================

# =============================================================================
#Question 3:

#INPUT:
Monthly_CRSP_Universe = PS2_Q2(Monthly_CRSP_Stocks,Monthly_CRSP_Bonds,Monthly_CRSP_Riskless)


#OUTPUT:

def PS2_Q3(Monthly_CRSP_Universe):    
    #different method, didnt work:
    #Monthly_CRSP_Universe['weight_stock'] = pd.rolling_std(Monthly_CRSP_Universe.Stock_Excess_Vw_Ret,window=36)

    #Calculate the value weighted return of the portfolio formed by Stocks and Bonds    
    Vw_wt_stock = Monthly_CRSP_Universe.Stock_lag_MV/(Monthly_CRSP_Universe.Stock_lag_MV+Monthly_CRSP_Universe.Bond_lag_MV)
    Vw_wt_bond  = Monthly_CRSP_Universe.Bond_lag_MV/(Monthly_CRSP_Universe.Stock_lag_MV+Monthly_CRSP_Universe.Bond_lag_MV)
    Monthly_CRSP_Universe['Excess_Vw_Ret'] = Vw_wt_stock*Monthly_CRSP_Universe.Stock_Excess_Vw_Ret + Vw_wt_bond*Monthly_CRSP_Universe.Bond_Excess_Vw_Ret

    #Calculate the returns of portfolio with 60% stocks and 40% bonds:
    Monthly_CRSP_Universe['Excess_60_40_Ret'] =  0.60*Monthly_CRSP_Universe.Stock_Excess_Vw_Ret + 0.40*Monthly_CRSP_Universe.Bond_Excess_Vw_Ret

    
    #Calculate inverse of rolling standard deviation with a window of three years prior with 1 period lag:
    Monthly_CRSP_Universe['Stock_inverse_sigma_hat'] = 1/(Monthly_CRSP_Universe.Stock_Excess_Vw_Ret.rolling(36).std()).shift(1)
    Monthly_CRSP_Universe['Bond_inverse_sigma_hat'] = 1/(Monthly_CRSP_Universe.Bond_Excess_Vw_Ret.rolling(36).std()).shift(1)
    
    #Calculating levered Portfolio:
    Monthly_CRSP_Universe['Unlevered_k'] = 1/(Monthly_CRSP_Universe['Bond_inverse_sigma_hat'] + Monthly_CRSP_Universe['Stock_inverse_sigma_hat'])
    
    Monthly_CRSP_Universe['weight_stock_unlevered'] =  Monthly_CRSP_Universe['Stock_inverse_sigma_hat']*Monthly_CRSP_Universe['Unlevered_k']
    Monthly_CRSP_Universe['weight_bond_unlevered']  =  Monthly_CRSP_Universe['Bond_inverse_sigma_hat']*Monthly_CRSP_Universe['Unlevered_k']
    Monthly_CRSP_Universe['Excess_Unlevered_RP_Ret'] = Monthly_CRSP_Universe['weight_stock_unlevered']*Monthly_CRSP_Universe.Stock_Excess_Vw_Ret+ Monthly_CRSP_Universe['weight_bond_unlevered']*Monthly_CRSP_Universe.Bond_Excess_Vw_Ret
    
    
    #Calculate Levered RP:
    #if k=1 then:
    A = Monthly_CRSP_Universe.Stock_Excess_Vw_Ret*Monthly_CRSP_Universe.Stock_inverse_sigma_hat + Monthly_CRSP_Universe.Bond_Excess_Vw_Ret*Monthly_CRSP_Universe.Bond_inverse_sigma_hat
    
    Monthly_CRSP_Universe['Levered_k'] = Monthly_CRSP_Universe.Excess_Vw_Ret.std()/A.std()
    
    Monthly_CRSP_Universe['weight_stock_Levered'] =  Monthly_CRSP_Universe['Stock_inverse_sigma_hat']*Monthly_CRSP_Universe['Levered_k']
    Monthly_CRSP_Universe['weight_bond_Levered']  =  Monthly_CRSP_Universe['Bond_inverse_sigma_hat']*Monthly_CRSP_Universe['Levered_k']
    Monthly_CRSP_Universe['Excess_Levered_RP_Ret'] = Monthly_CRSP_Universe['weight_stock_Levered']*Monthly_CRSP_Universe.Stock_Excess_Vw_Ret+ Monthly_CRSP_Universe['weight_bond_Levered']*Monthly_CRSP_Universe.Bond_Excess_Vw_Ret

    output_variables_PS2_Q3 = np.array(['Year','Month','Stock_Excess_Vw_Ret','Bond_Excess_Vw_Ret','Excess_Vw_Ret','Excess_60_40_Ret','Stock_inverse_sigma_hat','Bond_inverse_sigma_hat','Unlevered_k','Excess_Unlevered_RP_Ret','Levered_k','Excess_Levered_RP_Ret'])

    return(Monthly_CRSP_Universe[output_variables_PS2_Q3])

Q3_ans = PS2_Q3(Monthly_CRSP_Universe)
# =============================================================================



# =============================================================================
# Question 4:

#INPUT:
Port_Rets = PS2_Q3(Monthly_CRSP_Universe)

from scipy.stats import sem
#OUTPUT:

def PS2_Q4(Port_Rets):
    #keep data from Jan 1930 to June 2010:
    Port_Rets = Port_Rets[~((Port_Rets['Year'] >= 2010) & (Port_Rets['Month'] >= 7) )]
    Port_Rets = Port_Rets[~(Port_Rets['Year'] > 2010)]
    Port_Rets = Port_Rets[~(Port_Rets['Year'] < 1930)]


    d = {'CRSP Stocks': [Port_Rets.Stock_Excess_Vw_Ret.mean()*1200, Port_Rets.Stock_Excess_Vw_Ret.mean()/sem(Port_Rets.Stock_Excess_Vw_Ret),Port_Rets.Stock_Excess_Vw_Ret.std()*np.sqrt(12)*100,  Port_Rets.Stock_Excess_Vw_Ret.mean()*1200/(Port_Rets.Stock_Excess_Vw_Ret.std()*np.sqrt(12)*100),  Port_Rets.Stock_Excess_Vw_Ret.skew(), Port_Rets.Stock_Excess_Vw_Ret.kurtosis()],
         'CRSP Bonds' : [Port_Rets.Bond_Excess_Vw_Ret.mean()*1200, Port_Rets.Bond_Excess_Vw_Ret.mean()/sem(Port_Rets.Bond_Excess_Vw_Ret),Port_Rets.Bond_Excess_Vw_Ret.std()*np.sqrt(12)*100,  Port_Rets.Bond_Excess_Vw_Ret.mean()*1200/(Port_Rets.Bond_Excess_Vw_Ret.std()*np.sqrt(12)*100),  Port_Rets.Bond_Excess_Vw_Ret.skew(), Port_Rets.Bond_Excess_Vw_Ret.kurtosis()],
         'Value-weighted portfolio' : [Port_Rets.Excess_Vw_Ret.mean()*1200, Port_Rets.Excess_Vw_Ret.mean()/sem(Port_Rets.Excess_Vw_Ret),Port_Rets.Excess_Vw_Ret.std()*np.sqrt(12)*100,  Port_Rets.Excess_Vw_Ret.mean()*1200/(Port_Rets.Excess_Vw_Ret.std()*np.sqrt(12)*100),  Port_Rets.Excess_Vw_Ret.skew(), Port_Rets.Excess_Vw_Ret.kurtosis()],
         '60/40 portfolio': [Port_Rets.Excess_60_40_Ret.mean()*1200, Port_Rets.Excess_60_40_Ret.mean()/sem(Port_Rets.Excess_60_40_Ret),Port_Rets.Excess_60_40_Ret.std()*np.sqrt(12)*100,  Port_Rets.Excess_60_40_Ret.mean()*1200/(Port_Rets.Excess_60_40_Ret.std()*np.sqrt(12)*100),  Port_Rets.Excess_60_40_Ret.skew(), Port_Rets.Excess_60_40_Ret.kurtosis()],
         'unlevered RP': [Port_Rets.Excess_Unlevered_RP_Ret.mean()*1200, Port_Rets.Excess_Unlevered_RP_Ret.mean()/sem(Port_Rets.Excess_Unlevered_RP_Ret),Port_Rets.Excess_Unlevered_RP_Ret.std()*np.sqrt(12)*100,  Port_Rets.Excess_Unlevered_RP_Ret.mean()*1200/(Port_Rets.Excess_Unlevered_RP_Ret.std()*np.sqrt(12)*100),  Port_Rets.Excess_Unlevered_RP_Ret.skew(), Port_Rets.Excess_Unlevered_RP_Ret.kurtosis()],
         'levered RP': [Port_Rets.Excess_Levered_RP_Ret.mean()*1200, Port_Rets.Excess_Levered_RP_Ret.mean()/sem(Port_Rets.Excess_Levered_RP_Ret),Port_Rets.Excess_Levered_RP_Ret.std()*np.sqrt(12)*100,  Port_Rets.Excess_Levered_RP_Ret.mean()*1200/(Port_Rets.Excess_Levered_RP_Ret.std()*np.sqrt(12)*100),  Port_Rets.Excess_Levered_RP_Ret.skew(), Port_Rets.Excess_Levered_RP_Ret.kurtosis()]}
   
    Q4_ans = pd.DataFrame(data = d).T
    Q4_ans = Q4_ans.rename(columns = {0:'Annualized Mean',1:'t-Stat of Annualized Mean',2:'Annualized Standard Deviation',3:'Annualized Sharpe Ratio',4:'Skewness',5:'Excess Kurtosis'})
    
    return(Q4_ans)

Q4_ans = PS2_Q4(Port_Rets)

Q4_ans.to_excel("Q4_table.xlsx")
# =============================================================================









