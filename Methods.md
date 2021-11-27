---
layout: page
title: Supplementary Methods
permalink: /Methods/
---

__Supplementary methods__

*Python libraries and dependencies*

The following dependencies were used in the creation of this dashboard (see code snippet below)

<details>
<summary>Preview</summary>

<figure class="highlight">
    <pre>
        <code class="language-ruby" data-lang="ruby">
        <span class="nb">puts</span> <span class="s1">'Expanded message'</span>
        </code>
    </pre>
</figure>

*Data pre-processing*

Following anonymisation, referral data was uploaded as a pandas dataframe. Redundant columns, duplicates and erroneous entries were removed and all dates and times were transformed to python date-time data-types for further manipulation. Specialist working diagnoses are designated by the on-call neurosurgical registrar when receiving the referral and include a total of 138 different options. The diagnosis is based on the information received at the point of the referral and may be modified as further information is shared or after senior review. Specialist diagnoses were aggregated into 13 primary diagnostic categories: brain tumour, cauda equina syndrome, congenital, subdural haematoma, cranial trauma, degenerative spine, hydrocephalus, infection, spinal trauma, stroke, neurovascular and ‘not neurosurgical’ (Supplementary Appendix). 






