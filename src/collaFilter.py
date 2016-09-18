# -*- coding:utf-8 -*-

import math
import csv
import logging

# logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='../log/toutiao.log',
                    filemode='w')
logger = logging.getLogger('toutiao')
QUESTION_FILE = '../data/question_info.txt'
USERINFO_FILE = '../data/user_info.txt'
INVITEINFO_FILE = '../data/invited_info_train.txt'
VALIDATE_FILE = '../data/validate_nolabel.txt'
RESULT_FILE = '../data/temp.csv'


def readQuestionInfo():
    questionInfo = {}
    questionFile = open(QUESTION_FILE, 'r')
    maxLike = 0
    minLike = 1000000
    maxAnswer = 0
    minAnswer = 1000000
    maxEssence = 0
    minEssence = 1000000
    for line in questionFile:
        question = line.strip().split('\t')
        length = len(question)
        questionInfo[question[0]] = {
            'flag': None if length <= 1 else question[1],
            'words': None if length <= 2 else question[2],
            'chars': None if length <= 3 else question[3],
            'like': None if length <= 4 else int(question[4]),
            'answer': None if length <= 5 else int(question[5]),
            'essence': None if length <= 6 else int(question[6])
        }
        like = questionInfo[question[0]]['like']
        if like is not None:
            if like > maxLike:
                maxLike = like
            if like < minLike:
                minLike = like

        answer = questionInfo[question[0]]['answer']
        if answer is not None:
            if answer > maxAnswer:
                maxAnswer = answer
            if answer < maxAnswer:
                minAnswer = answer

        essence = questionInfo[question[0]]['essence']
        if essence is not None:
            if essence > maxEssence:
                maxEssence = essence
            if essence < minEssence:
                minEssence = essence
    questionInfo['maxLike'] = maxLike
    questionInfo['minLike'] = minLike
    questionInfo['maxAnswer'] = maxAnswer
    questionInfo['minAnswer'] = minAnswer
    questionInfo['maxEssence'] = maxEssence
    questionInfo['minEssence'] = minEssence
    return questionInfo


def readUserInfo():
    userInfo = {}
    userFile = open(USERINFO_FILE, 'r')
    for line in userFile:
        user = line.strip().split('\t')
        length = len(user)
        userInfo[user[0]] = {
            'flag': None if length <= 1 else user[1].strip().split('/'),
            'words': None if length <= 2 else user[2],
            'chars': None if length <= 3 else user[3]
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
    questions = []
    if userId in inviteInfo:
        questions = inviteInfo[userId]

    # calculate simarity between quesiton and user's desc and flag
    questionAndUserSim = computeSimWithUser(userInfo[userId], newQ)

    # calculate simarity between quesition and each question under the given user
    # meanwhile calculate the average score such as like, answer, and essence
    like = int(newQ['like'])
    answer = int(newQ['answer'])
    essence = int(newQ['essence'])

    averageQuestionSim = 0.0
    averageLike = 0.0
    averageAnswer = 0.0
    averageEssence = 0.0
    averageScore = 0.0

    count = 0
    for i in questions:
        eachQ = questionInfo[i]
        averageQuestionSim += computeSimarity(newQ, eachQ)
        averageLike += eachQ['like']
        averageAnswer += eachQ['answer']
        averageEssence += eachQ['essence']
        count += 1
    averageQuestionSim = 0 if count == 0 else averageQuestionSim / count
    averageLike = 0 if count == 0 else averageLike / count
    averageAnswer = 0 if count == 0 else averageAnswer / count
    averageEssence = 0 if count == 0 else averageEssence / count
    allLike = 1 if like + averageLike == 0 else like + averageLike
    allAnswer = 1 if answer + averageAnswer == 0 else answer + averageAnswer
    allEssence = 1 if essence + averageEssence == 0 else essence + averageEssence

    averageLikeScore = like / allLike if averageLike != 0 else 0
    averageAnswerScore = answer / allAnswer if averageAnswer != 0 else 0
    averageEssenceScore = essence / allEssence if averageEssence != 0 else 0

    averageScore = 0.1 * averageLikeScore \
        + 0.1 * averageAnswerScore + 0.1 * averageEssenceScore
    if questionAndUserSim != 0 and averageQuestionSim == 0:
        averageScore = 0.8 * questionAndUserSim
    elif questionAndUserSim == 0 and averageQuestionSim != 0:
        avargeScore = 0.8 * averageQuestionSim
    else:
        averageScore = 0.4 * (questionAndUserSim + averageQuestionSim)

    attrScore = computeSelfAttrScore(like, answer, essence, questionInfo)
    attrFinalScore = 0.4 * attrScore['likeScore'] + 0.3 * \
        attrScore['answerScore'] + 0.3 * attrScore['essenceScore']

    logger.info('questionId:%s, userId:%s, questionAndUserSim:%f, averageQuestionSim:%f, averageLikeScore:%f, \
    averageAnswerScore:%f, averageEssenceScore:%f, attrFinalScore:%f' % (questionId, userId, questionAndUserSim, averageQuestionSim, averageLikeScore, averageAnswerScore, averageEssenceScore, attrFinalScore))

    if averageScore == 0.0:
        averageScore = attrFinalScore
    else:
        averageScore += 0.2 * attrFinalScore
    logger.info('final Score:%f', averageScore)
    return averageScore


def computeSimWithUser(user, question):
    userFlag = user['flag']
    questionFlag = question['flag']
    sim = 0
    if questionFlag in userFlag:
        sim += 0.6

    userDesc = user['words']
    questionDesc = question['words']
    if userDesc is not None and questionDesc is not None:
        sim += 0.4 * cal_sentence_cosine_sim(userDesc, questionDesc,
                                             user['chars'], user['chars'])

    return sim


def computeSelfAttrScore(like, answer, essence, questionInfo):
    maxLike = questionInfo['maxLike']
    minLike = questionInfo['minLike']
    maxAnswer = questionInfo['maxAnswer']
    minAnswer = questionInfo['minAnswer']
    maxEssence = questionInfo['maxEssence']
    minEssence = questionInfo['minEssence']

    if like > 0 and like < 100:
        likeScore = 0.1
    elif like >= 100 and like < 200:
        likeScore = 0.2
    elif like >= 200 and like < 500:
        likeScore = 0.3
    elif like >= 500 and like < 1000:
        likeScore = 0.4
    elif like >= 1000:
        likeScore = 0.5
    else:
        likeScore = 0.0

    if answer > 0 and answer < 10:
        answerScore = 0.1
    elif answer >= 10 and answer < 20:
        answerScore = 0.2
    elif answer >= 20:
        answerScore = 0.5
    else:
        answerScore = 0.0

    if essence > 0 and essence < 10:
        essenceScore = 0.1
    elif essence > 10 and essence < 20:
        essenceScore = 0.2
    elif essence >= 20:
        essenceScore = 0.5
    else:
        essenceScore = 0.0

    return {
        'likeScore': likeScore,
        'answerScore': answerScore,
        'essenceScore': essenceScore
    }


def computeSimarity(question1, question2):
    score = 0
    if question1['flag'] is not None and question2['flag'] is not None and question1['flag'] == question2['flag']:
        score += 0.6
    if question1['words'] is not None and question2['words'] is not None:
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


def writeResultFile(resultMap):
    resultFile = open(RESULT_FILE, 'w+')
    minValue = 10.0
    maxValue = 0.0
    for record in resultMap:
        value = resultMap[record]
        if value < minValue:
            minValue = value
        if value > maxValue:
            maxValue = value
    print maxValue
    print minValue
    resultFile = open(RESULT_FILE, 'wb')
    writer = csv.writer(resultFile)
    writer.writerow(['qid', 'uid', 'label'])
    for record in resultMap:
        s = resultMap[record]
        s = s - minValue / maxValue - minValue
        newrecord = record.split(',')
        newrecord.append(s)
        writer.writerow(newrecord)


def validation():
    validateFile = open(VALIDATE_FILE, 'r')
    userInfo = readUserInfo()
    questionInfo = readQuestionInfo()
    inviteInfo = readInviteInfo()

    resultMap = {}

    count = 0
    for line in validateFile:
        if count == 0:
            count = 1
            continue
        record = line.strip().split(',')
        score = computeSimarities(record[0], record[1], questionInfo, userInfo, inviteInfo)
        resultMap[line.strip()] = score

    writeResultFile(resultMap)
if __name__ == '__main__':
    validation()
