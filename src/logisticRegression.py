# -*- coding:utf-8 -*-

from numpy import *
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
    for line in questionFile:
        question = line.strip().split('\t')
        length = len(question)
        questionInfo[question[0]] = {
            'flag': None if length <= 1 else question[1],
            'words': None if length <= 2 else question[2],
            'chars': None if length <= 3 else question[3],
            'like': 0 if length <= 4 else int(question[4]),
            'answer': 0 if length <= 5 else int(question[5]),
            'essence': 0 if length <= 6 else int(question[6])
        }
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
        inviteInfo[invite[1]].append({
            'qId': invite[0],
            'label': int(invite[2])
        })
    return inviteInfo


def calTagScore(user, question):
    userFlag = user['flag']
    questionFlag = question['flag']
    if questionFlag in userFlag:
        return 1.0
    return 0.0


def calWordsScore(user, question):
    userDesc = user['words']
    questionDesc = question['words']
    if userDesc is not None and questionDesc is not None:
        return _cal_sentence_cosine_sim(userDesc, questionDesc)
    return 0.0


def calCharsScore(user, question):
    userChars = user['chars']
    questionChars = question['chars']
    return _cal_chars_common_sim(userChars, questionChars)


def calLikeScore(like, maxLike, minLike):
    return float(like - minLike) / (maxLike - minLike)


def calAnswerScore(answer, maxAnswer, minAnswer):
    return float(answer - minAnswer) / (maxAnswer - minAnswer)


def calEssenceScore(essence, maxEssence, minEssence):
    return float(essence - minEssence) / (maxEssence - minEssence)


def _cal_sentence_cosine_sim(words1, words2):
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


def _cal_words_common_sim(words1, words2):
    return _cal_chars_common_sim(words1, words2)


def _cal_chars_common_sim(chars1, chars2):
    charsMap = {}
    chars1List = chars1.split('/')
    chars2List = chars2.split('/')
    for char in chars1List:
        charsMap[char] = 1

    for char in chars2List:
        if char not in charsMap:
            charsMap[char] = 1
        else:
            charsMap[char] = 2
    count = 0
    length = 0
    for char in charsMap:
        length += 1
        if charsMap[char] == 2:
            count += 1
    return float(count) / length


def sigmoid(intX):
    return 1.0 / (1 + exp(-intX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in xrange(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


def constuct(inviteInfo, userInfo, questionInfo):
    userModels = {}
    for userId in inviteInfo:
        user = userInfo[userId]
        qList = inviteInfo[userId]

        uMatrix = []
        labels = []
        for key in qList:
            qId = key['qId']
            label = key['label']
            question = questionInfo[qId]

            tagScore = calTagScore(user, question)
            wordsScore = calWordsScore(user, question)
            charsScore = calCharsScore(user, question)
            likeScore = question['like']
            answerScore = question['answer']
            essenceScore = question['essence']

            score = [tagScore, wordsScore, charsScore, likeScore, answerScore, essenceScore]
            uMatrix.append(score)
            labels.append(label)
        uMatrix = array(uMatrix, dtype=float)

        likeArray = [x[3] for x in uMatrix]
        answerArray = [x[4] for x in uMatrix]
        essenceArray = [x[5] for x in uMatrix]

        maxLike = max(likeArray)
        minLike = min(likeArray)
        maxAnswer = max(answerArray)
        minAnswer = min(answerArray)
        maxEssence = max(essenceArray)
        minEssence = min(essenceArray)

        for x in uMatrix:
            if maxLike == minLike:
                x[3] = 1 if maxLike != 0 else 0
            else:
                x[3] = (x[3] - minLike) / (maxLike - minLike)

            if maxAnswer == minAnswer:
                x[4] = 1 if maxAnswer != 0 else 0
            else:
                x[4] = (x[4] - minAnswer) / (maxAnswer - minAnswer)

            if maxEssence == minEssence:
                x[5] = 1 if maxEssence != 0 else 0
            else:
                x[5] = (x[5] - minEssence) / (maxEssence - minEssence)

        weights = gradAscent(uMatrix, labels)
        userModels[userId] = {
            'weights': weights,
            'maxLike': maxLike,
            'minLike': minLike,
            'maxAnswer': maxAnswer,
            'minAnswer': minAnswer,
            'maxEssence': maxEssence,
            'minEssence': minEssence
        }

    return userModels


def calUserTagSim(flag1, flag2):
    flagMap = {}
    for f in flag1:
        flagMap[f] = 1
    for f in flag2:
        if f in flag1:
            flagMap[f] += 1
        else:
            flagMap[f] = 1
    count = 0
    length = 0
    for f in flagMap:
        length += 1
        if flagMap[f] == 2:
            count += 1

    return float(count) / length


def calUserSim(user1, user2):
    words1 = user1['words']
    words2 = user2['words']
    chars1 = user1['chars']
    chars2 = user2['chars']
    flags1 = user1['flag']
    flags2 = user2['flag']
    wordsSim = _cal_words_common_sim(words1, words2)
    charsSim = _cal_chars_common_sim(chars1, chars2)
    tagSim = calUserTagSim(flags1, flags2)
    return 0.2 * wordsSim + 0.2 * charsSim + 0.6 * tagSim


def buildUserSimMatrix(userInfo):
    userSimMatrix = {}

    allusers = []
    for userId in userInfo:
        user = userInfo[userId]
        user['id'] = userId
        allusers.append(user)

    i = 0
    j = 0
    length = len(allusers)
    while i < length:
        ui = allusers[i]
        userSimMatrix[ui['id']] = []
        while j < length:
            if j != i:
                uj = allusers[j]
                userSim = calUserSim(ui, uj)
                userSimMatrix[ui['id']].append({
                    'id': uj['id'],
                    'score': userSim
                })
            j += 1
        i += 1
    return userSimMatrix


def getUserModels(userInfo, questionInfo, inviteInfo):
    return constuct(inviteInfo, userInfo, questionInfo)


def calQuestionScore(question, user, userModel):
    maxLike = userModel['maxLike']
    minLike = userModel['minLike']
    maxAnswer = userModel['maxAnswer']
    minAnswer = userModel['minAnswer']
    maxEssence = userModel['maxEssence']
    minEssence = userModel['minEssence']

    tagScore = calTagScore(user, question)
    wordsScore = calWordsScore(user, question)
    charsScore = calCharsScore(user, question)
    likeScore = 0.0
    answerScore = 0.0
    essenceScore = 0.0
    if maxLike != minLike:
        likeScore = float(question['like'] - minLike) / (maxLike - minLike)
    if maxAnswer != minAnswer:
        answerScore = float(question['answer'] - minAnswer) / (maxAnswer - minAnswer)
    if maxEssence != minEssence:
        essenceScore = float(question['essence'] - minEssence) / (maxEssence - minEssence)

    score = [tagScore, wordsScore, charsScore, likeScore, answerScore, essenceScore]
    return score


def findSimUser(qId, userId, userModels, userSimMatrix):
    simList = userSimMatrix[userId]
    sortedSim = {}

    for u in simList:
        sortedSim[u['score']] = u['id']

    userId = ''
    for eachkey in sorted(sortedSim):
        userId = sortedSim[eachkey]
        if userId in userModels:
            break
    return userId


def calRecommendProbability(qId, userId, questionInfo, userInfo, userModels, userSimMatrix):
    if userId not in userModels:
        userId = findSimUser(qId, userId, userModels, userSimMatrix)
    question = questionInfo[qId]
    user = userInfo[userId]
    model = userModels[userId]

    questionScore = calQuestionScore(question, user, model)
    weights = model['weights']
    probability = mat(questionScore) * mat(weights)
    return sigmoid(asarray(probability)[0][0])


def writeResultFile(resultMap):
    resultFile = open(RESULT_FILE, 'wb')
    writer = csv.writer(resultFile)
    writer.writerow(['qid', 'uid', 'label'])
    for record in resultMap:
        s = resultMap[record]
        newrecord = record.split(',')
        newrecord.append(s)
        writer.writerow(newrecord)


def validation():
    validateFile = open(VALIDATE_FILE, 'r')

    logger.info('read file...')
    userInfo = readUserInfo()
    questionInfo = readQuestionInfo()
    inviteInfo = readInviteInfo()
    logger.info('read file finished!')

    logger.info('build user sim matrix...')
    userSimMatrix = buildUserSimMatrix(userInfo)
    logger.info('build user sim matrix finished!')

    logger.info('construct user model...')
    userModels = getUserModels(userInfo, questionInfo, inviteInfo)
    logger.info('construct user model finished!')

    resultMap = {}
    count = 0

    for line in validateFile:
        if count == 0:
            count = 1
            continue
        record = line.strip().split(',')
        pro = calRecommendProbability(
            record[0], record[1], questionInfo, userInfo, userModels, userSimMatrix)
        resultMap[line.strip()] = pro
    writeResultFile(resultMap)

if __name__ == '__main__':
    validation()
