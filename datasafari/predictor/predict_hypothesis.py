def predict_hypothesis():
    """
    The general idea here is to create a function where:
    (1) Adheres to MVP of simplicity in user-input
    (2) In case of hypothesis it does everything automatically (e.g. for numeric data it will check
    for normality and choose parametric or non-parametric tests etc. it will also reject or not the null itself)
    (3) It therefore not only eases writing the code but also decision-making, the user does not need
    to know about: what test to use based on the data, what test of the appropriate tests are better (all are provided),
    and so on
    (4) It handles the key datatypes categorical and numeric automatically

    Schema of the idea:
    > In > user data (df and ??)
    | 1 Inside: infer numeric or categorical
    \\
     \\
        | 2A Inside: if numeric test assumptions to choose if parametric or non-parametric
        | 3A Inside: perform either set of tests and get results
        | 4A Inside: use results to build output and make a ready conclusion for the user
        < Out < results + conclusion
    \\
     \\
        | 2B Inside: if categorical  ...
        | 3B Inside: ...
        | 4B Inside: ...
        < Out < results + conclusion
    """
