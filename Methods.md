---
layout: page
title: Supplementary Methods
permalink: /Methods/
---

__Supplementary methods__

*Python libraries and dependencies*

The following dependencies were used in the creation of this dashboard (see code snippet below)

<details>
<summary>Code</summary>

{% highlight python %}
'pandas', 1.2.3
'numpy', 1.19.5
'matplotlib.pyplot', 3.4.1
'scipy', 1.6.2
'plotly', 5.3.1
'dash', 1.20.0
'dash_core_components', 1.16.0
'dash_html_components', 1.1.3
'requests', 2.25.1
'statsmodels', 0.11.0
'prophet', 1.0.1
'pmdarima' 1.81
'tensorflow' 2.4.1
{% endhighlight %}

</details>

*Data pre-processing*

Following anonymisation, referral data was uploaded as a pandas dataframe. Redundant columns, duplicates and erroneous entries were removed and all dates and times were transformed to python date-time data-types for further manipulation. Specialist working diagnoses are designated by the on-call neurosurgical registrar when receiving the referral and include a total of 138 different options. The diagnosis is based on the information received at the point of the referral and may be modified as further information is shared or after senior review. Specialist diagnoses were aggregated into 13 primary diagnostic categories: brain tumour, cauda equina syndrome, congenital, subdural haematoma, cranial trauma, degenerative spine, hydrocephalus, infection, spinal trauma, stroke, neurovascular and ‘not neurosurgical’ (Supplementary Appendix). 

<details>
<summary>Code</summary>

{% highlight python %}
#Upload anonymised file - either saved as .csv or .pkl

df_all = pd.read_pickle(filename)

#Drop duplicates
df_all.drop_duplicates(inplace=True)

#Drop redundant columns
df_all.drop(columns = ['Referring Doctor Name','Bleep or Telephone No','MobileNo','Subsequent Doctor Grade Name','Subsequent Bleep Number','Subsequent Mobile No','Subsequent Dr Email Address','Subsequent Consultant Email Address'], inplace = True)

#Transform date-time entries to datetime datatype
df_all = transform_to_datetime(df_all, 'Referral Time')

#Convert specialist working diagnosis into primary diagnostic classification based on diagnosis table - see Appendix table
diagnosis_table = pd.read_csv('diagnoses_table.csv', low_memory=False)
df_all = add_classification_level(df_all, diagnosis_table,
                                  'Primary Classification')


## RELEVANT PROCESSING FUNCTIONS

def match_classification(diagnosis_table, classification_level,
                               diagnosis):
    diagnosis_level = diagnosis_table[
        diagnosis_table['Specialist working diagnosis'] ==
        diagnosis][classification_level]
    if (len(diagnosis_level.values) > 0):
        return diagnosis_level.values[0]
    return 'no_match'

def add_classification(input_df, diagnosis_table, classification_level):
    df_copy = copy.deepcopy(input_df)
    partial_func = partial(match_classification, diagnosis_table,
                           classification_level)
    df_copy[classification_level] = df_copy[
        'Specialist Working Diagnosis'].apply(partial_func)
    return df_copy
    
def transform_to_datetime(df, time_col):
    copy = df.copy()
    copy[time_col] = pd.to_datetime(copy[time_col], dayfirst=True)
    return copy
{% endhighlight %}

</details>








