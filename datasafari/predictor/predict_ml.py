"""

The idea is to have a multi-use ML tool that is not as absolute as some of the modules in this package
and especially not like predict_hypothesis() which basically auto-performs the whole hypo. testing flow for the user.

Here we want the user to have an advanced ML toolkit which provides value in multiple ways.

The idea for predict_ml() was born through what I now will call 'model explorer', this is the first
toolkit in this module. It basically goes through various models and based on various scores and criteria
tells the user which model is best. It gives the model back and the user can further work with it.

If the user chooses to user predict_ml further then they can plug in this model they got from the 'model explorer'
and it can be used for training which would lead to production and insights.

Alternatively the module can be used for inference, where through model explorer we get the best model
and similarly to predict_hypothesis(), where we will provide the user with a model summary, interpretation, tips and conclusions.

So in summary currently it seems the functionality will be:
- 'model_recommender'
- 'model_tuner'
- 'auto_inference'
(or better names lol we'll see)

"""
