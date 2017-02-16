# Udacity-Project5

### Goal of this project

Enron was one of the largest companies in the United States,
By 2002, it had collapsed into bankruptcy due to widespread corporate fraud.
A significant amount of typically confidential information entered into the public record,
we are trying to building a person of interest identifier algorithm,
based on financial and email data made public as a result of the Enron scandal.

### Features and scaling

#### features

I create a function hist_and_save_pic which is now commented im poi_id.py, it will print and save
the feature histogram in png file. After explore all the features, I remove loan_advances,
restricted_stock_deferred, deferred_income, director_fees. Regarding the email features, I test
the correrlation between 'from_poi_to_this_person','from_this_person_to_poi' and 'shared_receipt_with_poi'
it's all highly related, so I only keep one feature from email features: 'shared_receipt_with_poi'.
I begin my test with decision tree classifier, as it turns out, only 'total_payments','restricted_stock_deferred', 'expenses'
are corrleated to precision and recall.


#### scaling

I didn't do any scaling, the scaling is not helpful for decision tree algorithm.

### new feature

