# -*- coding:utf-8 -*-

import logisticRegression as lr

if __name__ == '__main__':
    userInfo = lr.readUserInfo()
    questionInfo = lr.readQuestionInfo()
    userSimMap = lr.getSameUserMap('47e3b8d4204ae376b051dbfea2cbfc9f', userInfo)
    print userSimMap
