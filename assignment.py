import pandas as pd
import numpy as np

# Loading and Cleaning Energy Indicators Dataset 
xls_file = pd.ExcelFile('C:\\Users\\Admin\\Desktop\\Energy_Indicators.xls')
energy = xls_file.parse('Energy',skip_footer=(38), skiprows=17)
energy.drop(['Unnamed: 0', 'Unnamed: 2'], axis=1, inplace=True)
cols=['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
energy.columns = cols
energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']] = energy[['Energy Supply', 'Energy Supply per Capita', '% Renewable']].replace('...',np.NaN).apply(pd.to_numeric)
energy['Energy Supply'] = energy['Energy Supply'] * 1000000
energy['Country'] = energy['Country'].replace({'China, Hong Kong Special Administrative Region':'Hong Kong','United Kingdom of Great Britain and Northern Ireland':'United Kingdom','Republic of Korea':'South Korea','United States of America':'United States','Iran (Islamic Republic of)':'Iran'})
energy['Country'] = energy['Country'].str.replace(r" \(.*\)","")
energy

# GDP Dataset
GDP = pd.read_csv('C:\\Users\\Admin\\Desktop\\world_bank.csv', skiprows=4)
GDP['Country Name'] = GDP['Country Name'].replace({"Korea, Rep.": "South Korea", "Iran, Islamic Rep.": "Iran", "Hong Kong SAR, China": "Hong Kong"})
GDP = GDP[['Country Name','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']]
GDP.columns = ['Country','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']
GDP

# ScimEn
ScimEn = pd.read_excel(io='C:\\Users\\Admin\\Desktop\\scimagojr-3.xlsx')
ScimEn = ScimEn[:15]
ScimEn

# Merging Datasets
df = pd.merge(ScimEn,energy,how='inner',left_on='Country',right_on='Country')
df = pd.merge(df,GDP,how='inner',left_on='Country', right_on='Country')
df = df.set_index('Country')
column =  ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
df.columns = column
df

# Question 3
avgGDP = df[['2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']].mean(axis=1).rename('avgGDP').sort_values(ascending=False)
avgGDP

# Question 4
ans = df[df['Rank'] == 4]['2015'] - df[df['Rank'] == 4]['2006']
ans.tolist()[0]

# Question 5
ans = df['Energy Supply per Capita'].mean()
ans

# Question 6
ans = df[df['% Renewable'] == max(df['% Renewable'])]
ans = tuple((ans.index.tolist()[0],ans['% Renewable'].tolist()[0]))
ans

# Queston 7
df['Citation Ratio'] = df['Self-citations'] / df['Citations']
max_val = df['Citation Ratio'].max()
ans = df[df['Citation Ratio'] == max_val]
ans = tuple((ans.index.tolist()[0],max_val))
ans

# Question 8
df['Population'] = df['Energy Supply']/df['Energy Supply per Capita']
thid_most_populus = df.sort(columns='Population')['Population'].tolist()[-3]
ans = df[df['Population'] == thid_most_populus]
ans.index.tolist()[0]

# Question 10
df['High Renewable'] = [1 if x >= df['% Renewable'].median() else 0 for x in df['% Renewable']]
df

# Question 11
ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
df['PopEst'] = (df['Energy Supply'] / df['Energy Supply per Capita']).astype(float)
df = df.reset_index()
df['Continent'] = [ContinentDict[country] for country in df['Country']]
ans = df.set_index('Continent').groupby(level=0)['PopEst'].agg({'size': np.size, 'sum': np.sum, 'mean': np.mean,'std': np.std})
ans = ans[['size', 'sum', 'mean', 'std']]
ans