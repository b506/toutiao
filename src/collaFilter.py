# -*- coding:utf-8 -*-

import math
import logging

# logger
logging.basicConfig(level=logging.DEBUG,
                format='%(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='toutiao.log',
                filemode='w')
logger = logging.getLogger('toutiao')
QUESTION_FILE = '../data/question_info1.txt'
USERINFO_FILE = '../data/user_info1.txt'
INVITEINFO_FILE = '../data/invited_info_train1.txt'
VALIDATE_FILE = '../data/validate_nolabel1.txt'
RESULT_FILE = '../data/result.txt'


def readQuestionInfo():
    questionInfo = {}
    questionFile = open(QUESTION_FILE, 'r')
    for line in questionFile:
        question = line.strip().split('\t')
        questionInfo[question[0]] = {
            'flag': question[1],
            'words': question[2],
            'chars': question[3],
            'like': question[4],
            'answer': question[5],
            'essence': question[6]
        }
    return questionInfo


def readUserInfo():
    userInfo = {}
    userFile = open(USERINFO_FILE, 'r')
    for line in userFile:
        user = line.strip().split('\t')
        userInfo[user[0]] = {
            'flag': user[1].strip().split('/'),
            'words': user[2],
            'chars': user[3]
        }
    return userInfo


def readInviteInfo():
    inviteInfo = {}
    inviteFile = open(INVITEINFO_FILE, 'r')
    for line in inviteFile:
        invite = line.strip().split('\t')
        if invite[1] not in inviteInfo:
            inviteInfo[invite[1]] = []
        if int(invite[2]) == 1:
            inviteInfo[invite[1]].append(invite[0])
    return inviteInfo


def computeSimarities(questionId, userId, questionInfo, userInfo, inviteInfo):
    newQ = questionInfo[questionId]
    questions = inviteInfo[userId]

    # calculate simarity between quesiton and user's desc and flag
    questionAndUserSim = computeSimWithUser(userInfo[userId], newQ)

    # calculate simarity between quesition and each question under the given user
    # meanwhile calculate the average score such as like, answer, and essence
    like = int(newQ['like'])
    answer = int(newQ['answer'])
    essence = int(newQ['essence'])

    averageQuestionSim = 0
    averageLike = 0
    averageAnswer = 0
    averageEssence = 0
    averageScore = 0

    count = 0
    for i in questions:
        eachQ = questionInfo[i]
        averageQuestionSim += computeSimarity(newQ, eachQ)
        averageLike += int(eachQ['like'])
        averageAnswer += int(eachQ['answer'])
        averageEssence += int(eachQ['essence'])
        count += 1
    averageQuestionSim /= count
    averageLike /= count
    averageAnswer /= count
    averageEssence /= count
    allLike = 1 if like + averageLike == 0 else like + averageLike
    allAnswer = 1 if answer + averageAnswer == 0 else answer + averageAnswer
    allEssence = 1 if essence + averageEssence == 0 else essence + averageEssence

    # the final score is : 0.4* questionAndUserSim+ 0.3 * averageQuestionSim +0.1* (averageAnswer
    # +averageLike+ averageEssence)
    logger.info('questionId:%s, userId:%s, questionAndUserSim:%f, averageQuestionSim:%f, averageLike:%f, \
    averageAnswer:%f, averageEssence:%f'% (questionId, userId, questionAndUserSim, averageQuestionSim, float(like)/allLike\
    ,float(answer)/allAnswer, float(essence)/allEssence))

    averageScore = 0.4 * questionAndUserSim + 0.3 * averageQuestionSim + 0.1 * like / \
        allLike + 0.1 * answer / allAnswer + 0.1 * essence / allEssence
    return averageScore


def computeSimWithUser(user, question):
    userFlag = user['flag']
    questionFlag = question['flag']
    sim = 0
    if questionFlag in userFlag:
        sim += 0.6

    userDesc = user['words']
    questionDesc = question['words']
    if userDesc is not None:
        sim += 0.4 * cal_sentence_cosine_sim(userDesc, questionDesc,
                                             user['chars'], user['chars'])
    return sim


def computeSimarity(question1, question2):
    score = 0
    if question1['flag'] == question2['flag']:
        score += 0.6
    score += 0.4 * \
        cal_sentence_cosine_sim(question1['words'], question2[
                                'words'], question1['chars'], question2['chars'])
    return score


def cal_sentence_cosine_sim(words1, words2, chars1, chars2):
    sentence1 = words1.split('/')
    sentence2 = words2.split('/')
    map1 = {}
    map2 = {}
    for word in sentence1:
        if word not in map1:
            map1[word] = 1
        else:
            map1[word] += 1
    for word2 in sentence2:
        if word2 not in map2:
            map2[word2] = 1
        else:
            map2[word2] += 1
    allSum = 0
    denominator = 1

    for key1 in map1:
        if key1 in map2:
            allSum += map1[key1] * map2[key1]
            denominator *= math.sqrt(map1[key1] * map1[key1] + map2[key1] * map2[key1])
            map2[key1] = 0
        else:
            denominator *= math.sqrt(map1[key1] * map1[key1])

    for key2 in map2:
        if map2[key2] != 0:
            denominator *= math.sqrt(map2[key2] * map2[key2])

    return allSum / denominator


def validation():
    validateFile = open(VALIDATE_FILE, 'r')
    resultFile = open(RESULT_FILE, 'w+')
    userInfo = readUserInfo()
    questionInfo = readQuestionInfo()
    inviteInfo = readInviteInfo()

    count = 0
    for line in validateFile:
        if count == 0:
            count = 1
            continue
        record = line.strip().split(',')
        score = computeSimarities(record[0], record[1], questionInfo, userInfo, inviteInfo)
        resultFile.write(line.strip() + ',' + str(score) + '\r\n')
if __name__ == '__main__':
    validation()
