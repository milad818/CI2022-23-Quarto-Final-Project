# Free for personal or classroom use; see 'LICENSE.md' for details.
# https://github.com/squillero/computational-intelligence
import numpy
import numpy as np
from abc import abstractmethod
import copy
import random
from functools import cmp_to_key


class Player(object):

    def __init__(self, quarto, learningPhase, rounds=500, alpha=0.2, randomFactor=0.1) -> None:
        self.__quarto = quarto
        # self.__board = quarto.get_board_status
        self.historyOfMoves = []  # place, reward
        self.historyOfPiece = []  # piece, reward
        self.alpha = alpha
        self.randomFactor = randomFactor
        self.placeWeightDict = {}
        self.pieceWeightDict = {}
        self.learningPhase = learningPhase
        self.setReward(self.__quarto)
        self.rounds = rounds
        self.currentRound = 0
        self.tempSelected = set()

    def setReward(self, quarto):
        if self.learningPhase:
            for i, row in enumerate(quarto.get_board_status()):
                for j, col in enumerate(row):
                    self.placeWeightDict[(i, j)] = np.random.uniform(low=1.0, high=0.1)

            for pI,_ in enumerate(quarto.get_all_pieces()):
                self.pieceWeightDict[pI] = np.random.uniform(low=1.0, high=0.1)


    @abstractmethod
    def choose_piece(self) -> int:
        # self.__quarto.boa
        self.__quarto.getAvailablePieces()
        maxG = -10e15
        maxGrule = -10e9

        next_piece_index = None
        randomN = np.random.random()
        if self.learningPhase:

            if randomN < self.randomFactor:
                next_piece_index = random.randint(0, 15)

            else:
                pieceWeight = []
                for pieceIndex in self.__quarto.availablePieces:
                    selectedPieceWeight = self.__quarto.calcPieceWeight(pieceIndex)
                    pieceWeight.append((pieceIndex, selectedPieceWeight))

                sortedList = sorted(pieceWeight, key=cmp_to_key(self.compare), reverse=True)
                # the worst
                next_piece_index_help, _ = sortedList[0]
                # next_piece_index = next_piece_index_help

                for pieceIndex, weight in pieceWeight:
                    new_piece_index = pieceIndex
                    tempW = weight
                    if self.pieceWeightDict[new_piece_index] >= maxG or tempW >= maxGrule:
                        next_piece_index = new_piece_index
                        maxG = self.pieceWeightDict[pieceIndex]
                        maxGrule = tempW

                if next_piece_index is None:
                    print("help piece")
                    next_piece_index = next_piece_index_help



        else:
            for pieceIndex in self.__quarto.availablePieces:
                new_piece_index = pieceIndex
                if self.pieceWeightDict[new_piece_index] >= maxG:
                    next_piece_index = new_piece_index
                    maxG = self.pieceWeightDict[pieceIndex]

        return next_piece_index

    def update_player_board(self):
        self.__board = self.__quarto.get_board_status

    def compare(self, pair1, pair2):
        _, fitness1 = pair1
        _, fitness2 = pair2

        if fitness2 > fitness1:
            return -1
        else:
            return 1

    @abstractmethod
    def place_piece(self) -> tuple[int, int]:
        # self.__board= self.__quarto.get_board_status()
        pI = self.__quarto.get_selected_piece()
        self.__quarto.updatePlayedPiecePlace()
        self.__quarto.getFreePlaces()
        maxG = -10e15
        maxGrun = -10e15
        next_move = None
        randomN = np.random.random()

        if self.learningPhase:

            if randomN < self.randomFactor:
                freePLaces = [tuple(i) for i in self.__quarto.allFreePlaces]
                next_move = random.choice(freePLaces)
            else:
                pairXYfeasibility = []
                for placeXY in self.__quarto.allFreePlaces:
                    weight = self.__quarto.calcPlaceWeight(pI, placeXY)
                    pairXYfeasibility.append((placeXY, weight))
                sortedList = sorted(pairXYfeasibility, key=cmp_to_key(self.compare), reverse=True)
                next_move_help, _ = sortedList[0]
                next_move = next_move_help

                for action, palceWeight in pairXYfeasibility:
                    new_state = action
                    # and palceWeight >= maxGrun
                    if self.placeWeightDict[new_state] >= maxG and palceWeight >= maxGrun:
                        next_move = new_state
                        maxG = self.placeWeightDict[new_state]
                        maxGrun = palceWeight

                if next_move is None:
                    print("place help")
                    next_move = next_move_help

        else:
            for action in self.__quarto.allFreePlaces:
                new_state = action
                if self.placeWeightDict[new_state] >= maxG:
                    next_move = new_state
                    maxG = self.placeWeightDict[new_state]
        return next_move

    def get_game(self):
        return self.__quarto

    # def updateMovesHistory(self, place):
    #     if self.__quarto.assignReward() != 0:
    #         reward = 1 / self.__quarto.assignReward()
    #     else:
    #         reward =  self.__quarto.assignReward()
    #     self.historyOfMoves.append((place, reward))

    def updateHistoryOfMoves(self, place):
        reward = self.__quarto.assignReward()
        self.historyOfMoves.append((place, reward))


    def updateHistoryOfPiece(self, piece):
        reward = self.__quarto.assignReward()
        self.historyOfPiece.append((piece, reward))


class CharacCounter(object):
    def __init__(self, selectedPieceCharacteristic = None):
        if selectedPieceCharacteristic == None:
            self.high = 0
            self.notHigh = 0
            self.colored = 0
            self.notColored = 0
            self.solid = 0
            self.notSolid = 0
            self.square = 0
            self.notSquare = 0
        else:
            if selectedPieceCharacteristic.HIGH:
                self.high = 1
                self.notHigh = 0
            else:
                self.notHigh = 1
                self.high = 0

            if selectedPieceCharacteristic.COLOURED:
                self.colored = 1
                self.notColored = 0
            else:
                self.notColored = 1
                self.colored = 0

            if selectedPieceCharacteristic.SOLID:
                self.solid = 1
                self.notSolid = 0
            else:
                self.notSolid = 1
                self.solid = 0

            if selectedPieceCharacteristic.SQUARE:
                self.square = 1
                self.notSquare = 0
            else:
                self.notSquare = 1
                self.square = 0

class Piece(object):

    def __init__(self, high: bool, coloured: bool, solid: bool, square: bool) -> None:
        self.HIGH = high
        self.COLOURED = coloured
        self.SOLID = solid
        self.SQUARE = square


class Quarto(object):

    MAX_PLAYERS = 2
    BOARD_SIDE = 4

    def __init__(self) -> None:
        self.__players = ()
        self.reset()


    def reset(self):
        self.__board = np.ones(shape=(self.BOARD_SIDE, self.BOARD_SIDE), dtype=int) * -1
        self.__pieces = []
        self.__pieces.append(Piece(False, False, False, False))  # 0
        self.__pieces.append(Piece(False, False, False, True))  # 1
        self.__pieces.append(Piece(False, False, True, False))  # 2
        self.__pieces.append(Piece(False, False, True, True))  # 3
        self.__pieces.append(Piece(False, True, False, False))  # 4
        self.__pieces.append(Piece(False, True, False, True))  # 5
        self.__pieces.append(Piece(False, True, True, False))  # 6
        self.__pieces.append(Piece(False, True, True, True))  # 7
        self.__pieces.append(Piece(True, False, False, False))  # 8
        self.__pieces.append(Piece(True, False, False, True))  # 9
        self.__pieces.append(Piece(True, False, True, False))  # 10
        self.__pieces.append(Piece(True, False, True, True))  # 11
        self.__pieces.append(Piece(True, True, False, False))  # 12
        self.__pieces.append(Piece(True, True, False, True))  # 13
        self.__pieces.append(Piece(True, True, True, False))  # 14
        self.__pieces.append(Piece(True, True, True, True))  # 15
        self.__current_player = 0
        self.__selected_piece_index = -1
        self.getFreePlaces()
        self.learningPhase = False
        self.allFreePlaces = None
        self.availablePieces = None

    def set_players(self, players: tuple[Player, Player]):
        self.__players = players

    def updatePlayedPiecePlace(self):
        self.getAvailablePieces()
        self.getFreePlaces()

    def getAvailablePieces(self):
        allIndices = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        board = self.get_board_status().ravel()
        allSelectedPieces = set(board[board > -1])
        self.availablePieces = list(allIndices - allSelectedPieces)

    def getFreePlaces(self):
        self.freePlaces = np.where(self.__board == -1)
        self.allFreePlaces = zip(self.freePlaces[0], self.freePlaces[1])


    def compare(self, pair1, pair2):
        _, fitness1 = pair1
        _, fitness2 = pair2

        if fitness2 > fitness1:
            return -1
        else:
            return 1

    def select(self, pieceIndex: int) -> bool:
        '''
        select a piece. Returns True on success
        '''
        if pieceIndex not in self.__board:
            self.__selected_piece_index = pieceIndex
            return True
        return False

    def place(self, x: int, y: int) -> bool:
        '''
        Place piece in coordinates (x, y). Returns true on success
        '''
        if self.__placeable(x, y):
            self.__board[x, y] = self.__selected_piece_index
            return True
        return False

    def __placeable(self, x: int, y: int) -> bool:
        return not (y < 0 or x < 0 or x > 3 or y > 3 or self.__board[x, y] >= 0)

    def print(self):
        '''
        Print the __board
        '''
        for row in self.__board:
            print("\n -------------------")
            print("|", end="")
            for element in row:
                print(f" {element: >2}", end=" |")
        print("\n -------------------\n")
        print(f"Selected piece: {self.__selected_piece_index}\n")

    def get_piece_charachteristics(self, index: int) -> Piece:
        '''
        Gets charachteristics of a piece (index-based)
        '''
        return copy.deepcopy(self.__pieces[index])

    def get_board_status(self) -> np.ndarray:
        '''
        Get the current __board status (__pieces are represented by index)
        '''
        return copy.deepcopy(self.__board)

    def get_all_pieces(self):
        return copy.deepcopy(self.__pieces)

    def get_selected_piece(self) -> int:
        '''
        Get index of selected piece
        '''
        return copy.deepcopy(self.__selected_piece_index)

    def __check_horizontal(self) -> int:
        hDict = dict()
        hDict[0] = None
        hDict[1] = None
        hDict[2] = None
        hDict[3] = None

        for i in range(self.BOARD_SIDE):
            initList = CharacCounter()

            high_values = [
                elem for elem in self.__board[i] if elem >= 0 and self.__pieces[elem].HIGH
            ]
            initList.high = len(high_values)

            coloured_values = [
                elem for elem in self.__board[i] if elem >= 0 and self.__pieces[elem].COLOURED
            ]
            initList.colored = len(coloured_values)

            solid_values = [
                elem for elem in self.__board[i] if elem >= 0 and self.__pieces[elem].SOLID
            ]
            initList.solid = len(solid_values)

            square_values = [
                elem for elem in self.__board[i] if elem >= 0 and self.__pieces[elem].SQUARE
            ]
            initList.square = len(square_values)

            low_values = [
                elem for elem in self.__board[i] if elem >= 0 and not self.__pieces[elem].HIGH
            ]
            initList.notHigh = len(low_values)

            noncolor_values = [
                elem for elem in self.__board[i] if elem >= 0 and not self.__pieces[elem].COLOURED
            ]
            initList.notColored = len(noncolor_values)

            hollow_values = [
                elem for elem in self.__board[i] if elem >= 0 and not self.__pieces[elem].SOLID
            ]
            initList.notSolid = len(hollow_values)

            circle_values = [
                elem for elem in self.__board[i] if elem >= 0 and not self.__pieces[elem].SQUARE
            ]
            initList.notSquare = len(circle_values)
            hDict[i] = initList

            if len(high_values) == self.BOARD_SIDE or len(
                    coloured_values
            ) == self.BOARD_SIDE or len(solid_values) == self.BOARD_SIDE or len(
                square_values) == self.BOARD_SIDE or len(low_values) == self.BOARD_SIDE or len(
                noncolor_values) == self.BOARD_SIDE or len(
                hollow_values) == self.BOARD_SIDE or len(
                circle_values) == self.BOARD_SIDE:
                return self.__current_player, None

        return -1, hDict

    def __check_vertical(self):
        vDict = dict()
        vDict[0] = None
        vDict[1] = None
        vDict[2] = None
        vDict[3] = None

        for i in range(self.BOARD_SIDE):
            # counts the total value of hight are selected
            initList = CharacCounter()
            high_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and self.__pieces[elem].HIGH
            ]
            initList.high = len(high_values)

            coloured_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and self.__pieces[elem].COLOURED
            ]
            initList.colored = len(coloured_values)

            solid_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and self.__pieces[elem].SOLID
            ]
            initList.solid = len(solid_values)

            square_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and self.__pieces[elem].SQUARE
            ]
            initList.square = len(square_values)

            low_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and not self.__pieces[elem].HIGH
            ]
            initList.notHigh = len(low_values)

            noncolor_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and not self.__pieces[elem].COLOURED
            ]
            initList.notColored = len(noncolor_values)

            hollow_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and not self.__pieces[elem].SOLID
            ]
            initList.notSolid = len(hollow_values)

            circle_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and not self.__pieces[elem].SQUARE
            ]
            initList.notSquare = len(circle_values)

            vDict[i] = initList

            if len(high_values) == self.BOARD_SIDE or len(
                    coloured_values
            ) == self.BOARD_SIDE or len(solid_values) == self.BOARD_SIDE or len(
                square_values) == self.BOARD_SIDE or len(low_values) == self.BOARD_SIDE or len(
                noncolor_values) == self.BOARD_SIDE or len(
                hollow_values) == self.BOARD_SIDE or len(
                circle_values) == self.BOARD_SIDE:
                return self.__current_player, None
        return -1, vDict

    def new_check_diagonal(self):
        LToRdiagDict = dict()
        LToRdiagDict[0] = None
        LToRdiagDict[1] = None
        LToRdiagDict[2] = None
        LToRdiagDict[3] = None

        for i in range(self.BOARD_SIDE):
            # if self.__board[i, i] < 0:
            #     break
            high_values = []
            coloured_values = []
            solid_values = []
            square_values = []
            low_values = []
            noncolor_values = []
            hollow_values = []
            circle_values = []
            LTRinitiallist = CharacCounter()

            if self.__pieces[self.__board[i, i]].HIGH:
                if self.__board[i, i] != -1:
                    high_values.append(self.__board[i, i])
                    LTRinitiallist.high = len(high_values)
            else:
                if self.__board[i, i] != -1:
                    low_values.append(self.__board[i, i])
                    LTRinitiallist.notHigh = len(low_values)

            if self.__pieces[self.__board[i, i]].COLOURED:
                if self.__board[i, i] != -1:
                    coloured_values.append(self.__board[i, i])
                    LTRinitiallist.colored = len(coloured_values)
            else:
                if self.__board[i, i] != -1:
                    noncolor_values.append(self.__board[i, i])
                    LTRinitiallist.notColored = len(noncolor_values)

            if self.__pieces[self.__board[i, i]].SOLID:
                if self.__board[i, i] != -1:
                    solid_values.append(self.__board[i, i])
                    LTRinitiallist.solid = len(solid_values)
            else:
                if self.__board[i, i] != -1:
                    hollow_values.append(self.__board[i, i])
                    LTRinitiallist.notSolid = len(hollow_values)

            if self.__pieces[self.__board[i, i]].SQUARE:
                if self.__board[i, i] != -1:
                    square_values.append(self.__board[i, i])
                    LTRinitiallist.square = len(square_values)
            else:
                if self.__board[i, i] != -1:
                    circle_values.append(self.__board[i, i])
                    LTRinitiallist.notSquare = len(circle_values)

            LToRdiagDict[i] = LTRinitiallist

        if len(high_values) == self.BOARD_SIDE or len(coloured_values) == self.BOARD_SIDE or len(
                solid_values) == self.BOARD_SIDE or len(square_values) == self.BOARD_SIDE or len(
            low_values
        ) == self.BOARD_SIDE or len(noncolor_values) == self.BOARD_SIDE or len(
            hollow_values) == self.BOARD_SIDE or len(circle_values) == self.BOARD_SIDE:
            return self.__current_player, None

        RToLdiagDict = dict()
        RToLdiagDict[0] = None
        RToLdiagDict[1] = None
        RToLdiagDict[2] = None
        RToLdiagDict[3] = None

        for i in range(self.BOARD_SIDE):
            # if self.__board[i, self.BOARD_SIDE - 1 - i] < 0:
            #     break
            high_values = []
            coloured_values = []
            solid_values = []
            square_values = []
            low_values = []
            noncolor_values = []
            hollow_values = []
            circle_values = []
            RTLinitiallist = CharacCounter()
            if self.__pieces[self.__board[i, self.BOARD_SIDE - 1 - i]].HIGH:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    high_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.high = len(high_values)
            else:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    low_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.notHigh = len(low_values)

            if self.__pieces[self.__board[i, self.BOARD_SIDE - 1 - i]].COLOURED:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    coloured_values.append(
                        self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.colored = len(coloured_values)
            else:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    noncolor_values.append(
                        self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.notColored = len(noncolor_values)

            if self.__pieces[self.__board[i, self.BOARD_SIDE - 1 - i]].SOLID:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    solid_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.solid = len(solid_values)
            else:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    hollow_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.notSolid = len(hollow_values)

            if self.__pieces[self.__board[i, self.BOARD_SIDE - 1 - i]].SQUARE:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    square_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.square = len(square_values)
            else:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    circle_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.notSquare = len(circle_values)

            RToLdiagDict[i] = RTLinitiallist

        if len(high_values) == self.BOARD_SIDE or len(coloured_values) == self.BOARD_SIDE or len(
                solid_values) == self.BOARD_SIDE or len(square_values) == self.BOARD_SIDE or len(
            low_values
        ) == self.BOARD_SIDE or len(noncolor_values) == self.BOARD_SIDE or len(
            hollow_values) == self.BOARD_SIDE or len(circle_values) == self.BOARD_SIDE:
            return self.__current_player, None

        retunDict = {
            "LTR": LToRdiagDict,
            "RTL": RToLdiagDict
        }
        return -1, retunDict

    def calcPieceWeight(self, pieceIndex):
        badPrise = -10e10
        copyOfBoard = copy.deepcopy(self.__board)
        self.updatePlayedPiecePlace()
        thisFreePlacesRewardForGivenPiece = []
        for possition in self.allFreePlaces:
            x, y = possition
            self.__board = copy.deepcopy(copyOfBoard)
            self.__board[x, y] = pieceIndex
            _, dictH = self.__check_horizontal()
            _, dictV = self.__check_vertical()
            _, dictD = self.new_check_diagonal()
            dictLTR, dictRTL = dictD["LTR"], dictD["RTL"]
            if dictH is not None and dictV is not None and dictLTR is not None and dictRTL is not None:
                propertyH = dictH[x]
                propertyV = dictV[y]
                thisStepH = numpy.array(
                    [propertyH.high, propertyH.notHigh, propertyH.colored, propertyH.notColored, propertyH.solid,
                     propertyH.notSolid, propertyH.square,
                     propertyH.notSquare])
                thisStepV = numpy.array(
                    [propertyV.high, propertyV.notHigh, propertyV.colored, propertyV.notColored, propertyV.solid,
                     propertyV.notSolid, propertyV.square,
                     propertyV.notSquare])
                l1 = dictLTR[0]
                l2 = dictLTR[1]
                l3 = dictLTR[2]
                l4 = dictLTR[3]
                thisStepL = numpy.array(
                    [l1.high + l2.high + l3.high + l4.high, l1.notHigh + l2.notHigh + l3.notHigh + l4.notHigh,
                     l1.colored + l2.colored + l3.colored + l4.colored,
                     l1.notColored + l2.notColored + l3.notColored + l4.notColored,
                     l1.solid + l2.solid + l3.solid + l4.solid, l1.notSolid + l2.notSolid + l3.notSolid + l4.notSolid,
                     l1.square + l2.square + l3.square + l4.square,
                     l1.notSquare + l2.notSquare + l3.notSquare + l4.notSquare])

                r1 = dictRTL[0]
                r2 = dictRTL[1]
                r3 = dictRTL[2]
                r4 = dictRTL[3]
                thisStepR = numpy.array(
                    [r1.high + r2.high + r3.high + r4.high, r1.notHigh + r2.notHigh + r3.notHigh + r4.notHigh,
                     r1.colored + r2.colored + r3.colored + r4.colored,
                     r1.notColored + r2.notColored + r3.notColored + r4.notColored,
                     r1.solid + r2.solid + r3.solid + r4.solid, r1.notSolid + r2.notSolid + r3.notSolid + r4.notSolid,
                     r1.square + r2.square + r3.square + r4.square,
                     r1.notSquare + r2.notSquare + r3.notSquare + r4.notSquare])

                h4 = len(thisStepH[thisStepH == 4])
                h3 = len(thisStepH[thisStepH == 3])
                h2 = len(thisStepH[thisStepH == 2])
                h1 = len(thisStepH[thisStepH == 1])
                h0 = len(thisStepH[thisStepH == 0])

                v4 = len(thisStepV[thisStepV == 4])
                v3 = len(thisStepV[thisStepV == 3])
                v2 = len(thisStepV[thisStepV == 2])
                v1 = len(thisStepV[thisStepV == 1])
                v0 = len(thisStepV[thisStepV == 0])

                l4 = len(thisStepL[thisStepL == 4])
                l3 = len(thisStepL[thisStepL == 3])
                l2 = len(thisStepL[thisStepL == 2])
                l1 = len(thisStepL[thisStepL == 1])
                l0 = len(thisStepL[thisStepL == 0])

                r4 = len(thisStepR[thisStepR == 4])
                r3 = len(thisStepR[thisStepR == 3])
                r2 = len(thisStepR[thisStepR == 2])
                r1 = len(thisStepR[thisStepR == 1])
                r0 = len(thisStepR[thisStepR == 0])

                temp = [h4, h3, h2, h1, h0, v4, v3, v2, v1, v0, l4, l3, l2, l1, l0, r4, r3, r2, r1, r0]
                forNormalize = numpy.array(temp)
                forNormalize = forNormalize[forNormalize != 0]
                totalValued = sum(forNormalize)

                if r4 != 0 or l4 != 0 or v4 != 0 or h4 != 0:
                    thisFreePlacesRewardForGivenPiece.append(((x, y), badPrise))
                else:
                    newPrise = 5 * (h3 + v3 + l3 + r3) + 2 * (h2 + v2 + l2 + r2) + 1 * (h1 + v1 + l1 + r1) + 1 * (
                            h0 + v0 + l0 + r0)

                    newPrise = newPrise / totalValued
                    thisFreePlacesRewardForGivenPiece.append(((x, y), newPrise))
            else:
                self.__board = copy.deepcopy(copyOfBoard)
                return badPrise

        sortedList = sorted(thisFreePlacesRewardForGivenPiece, key=cmp_to_key(self.compare), reverse=True)
        worstPlaceForGivenPiece, worstPriseForGivenPiece = sortedList[0]
        self.__board = copy.deepcopy(copyOfBoard)
        return worstPriseForGivenPiece

    def __check_diagonal(self):
        LTRdiagDict = dict()
        LTRdiagDict[0] = None
        LTRdiagDict[1] = None
        LTRdiagDict[2] = None
        LTRdiagDict[3] = None
        high_values = []
        coloured_values = []
        solid_values = []
        square_values = []
        low_values = []
        noncolor_values = []
        hollow_values = []
        circle_values = []

        for i in range(self.BOARD_SIDE):
            # if self.__board[i, i] < 0:
            #     break
            LTRinitiallist = CharacCounter()

            if self.__pieces[self.__board[i, i]].HIGH:
                if self.__board[i, i] != -1:
                    high_values.append(self.__board[i, i])
                    LTRinitiallist.high = len(high_values)
            else:
                if self.__board[i, i] != -1:
                    low_values.append(self.__board[i, i])
                    LTRinitiallist.notHigh = len(low_values)

            if self.__pieces[self.__board[i, i]].COLOURED:
                if self.__board[i, i] != -1:
                    coloured_values.append(self.__board[i, i])
                    LTRinitiallist.colored = len(coloured_values)
            else:
                if self.__board[i, i] != -1:
                    noncolor_values.append(self.__board[i, i])
                    LTRinitiallist.notColored = len(noncolor_values)

            if self.__pieces[self.__board[i, i]].SOLID:
                if self.__board[i, i] != -1:
                    solid_values.append(self.__board[i, i])
                    LTRinitiallist.solid = len(solid_values)
            else:
                if self.__board[i, i] != -1:
                    hollow_values.append(self.__board[i, i])
                    LTRinitiallist.notSolid = len(hollow_values)

            if self.__pieces[self.__board[i, i]].SQUARE:
                if self.__board[i, i] != -1:
                    square_values.append(self.__board[i, i])
                    LTRinitiallist.square = len(square_values)
            else:
                if self.__board[i, i] != -1:
                    circle_values.append(self.__board[i, i])
                    LTRinitiallist.notSquare = len(circle_values)

            LTRdiagDict[i] = LTRinitiallist

        if len(high_values) == self.BOARD_SIDE or len(coloured_values) == self.BOARD_SIDE or len(
                solid_values) == self.BOARD_SIDE or len(square_values) == self.BOARD_SIDE or len(
            low_values
        ) == self.BOARD_SIDE or len(noncolor_values) == self.BOARD_SIDE or len(
            hollow_values) == self.BOARD_SIDE or len(circle_values) == self.BOARD_SIDE:
            return self.__current_player, None

        RTLdiagDict = dict()
        RTLdiagDict[0] = None
        RTLdiagDict[1] = None
        RTLdiagDict[2] = None
        RTLdiagDict[3] = None
        high_values = []
        coloured_values = []
        solid_values = []
        square_values = []
        low_values = []
        noncolor_values = []
        hollow_values = []
        circle_values = []

        for i in range(self.BOARD_SIDE):
            # if self.__board[i, self.BOARD_SIDE - 1 - i] < 0:
            #     break
            RTLinitiallist = CharacCounter()
            if self.__pieces[self.__board[i, self.BOARD_SIDE - 1 - i]].HIGH:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    high_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.high = len(high_values)
            else:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    low_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.notHigh = len(low_values)

            if self.__pieces[self.__board[i, self.BOARD_SIDE - 1 - i]].COLOURED:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    coloured_values.append(
                        self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.colored = len(coloured_values)
            else:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    noncolor_values.append(
                        self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.notColored = len(noncolor_values)

            if self.__pieces[self.__board[i, self.BOARD_SIDE - 1 - i]].SOLID:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    solid_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.solid = len(solid_values)
            else:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    hollow_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.notSolid = len(hollow_values)

            if self.__pieces[self.__board[i, self.BOARD_SIDE - 1 - i]].SQUARE:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    square_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.square = len(square_values)
            else:
                if self.__board[i, self.BOARD_SIDE - 1 - i] != -1:
                    circle_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
                    RTLinitiallist.notSquare = len(circle_values)

            RTLdiagDict[i] = RTLinitiallist

        if len(high_values) == self.BOARD_SIDE or len(coloured_values) == self.BOARD_SIDE or len(
                solid_values) == self.BOARD_SIDE or len(square_values) == self.BOARD_SIDE or len(
            low_values
        ) == self.BOARD_SIDE or len(noncolor_values) == self.BOARD_SIDE or len(
            hollow_values) == self.BOARD_SIDE or len(circle_values) == self.BOARD_SIDE:
            return self.__current_player, None

        retunDict = {
            "LTR": LTRdiagDict,
            "RTL": RTLdiagDict
        }
        return -1, retunDict


    def assignReward(self):
        if not self.check_finished():
            return -100 * int(not self.check_finished())
        else:
            return 10e2

    def check_winner(self) -> int:
        '''
        Check who is the winner
        '''

        checkV, _ = self.__check_vertical()
        checkH, _ = self.__check_horizontal()
        checkD, _ = self.__check_diagonal()

        l = [checkH, checkV, checkD]
        # l = [self.__check_horizontal(), self.__check_vertical(), self.__check_diagonal()]
        for elem in l:
            if elem >= 0:
                return elem
        return -1

    def check_finished(self) -> bool:
        '''
        Check who is the loser
        '''
        for row in self.__board:
            for elem in row:
                if elem == -1:
                    return False
        return True

    def learnModelParams(self, pieces):
        agentRL = 0
        agentRandom = 0
        draw = 0
        if self.__players[0].learningPhase:
            print("inlearing ")

            for epoch in range(self.__players[0].rounds):
                self.__board = np.ones(shape=(self.BOARD_SIDE, self.BOARD_SIDE), dtype=int) * -1
                self.availablePieces = pieces
                winner = -1

                ++self.__players[0].currentRound
                while winner < 0 and not self.check_finished():

                    self.updatePlayedPiecePlace()
                    piece_ok = False
                    while not piece_ok:
                        self.updatePlayedPiecePlace()
                        selectedPiece = self.__players[self.__current_player].choose_piece()
                        piece_ok = self.select(selectedPiece)
                        if self.__players[0].learningPhase:
                            if piece_ok and not bool(self.__current_player):
                                self.__players[0].updateHistoryOfPiece(self.__selected_piece_index)
                    piece_ok = False
                    self.__current_player = (self.__current_player + 1) % self.MAX_PLAYERS
                    while not piece_ok:
                        self.updatePlayedPiecePlace()
                        place = self.__players[self.__current_player].place_piece()
                        x, y = place
                        piece_ok = self.place(x, y)
                        if self.__players[0].learningPhase:
                            if piece_ok and not bool(self.__current_player):
                                self.__players[0].updateHistoryOfMoves(place)
                    winner = self.check_winner()
                print(f"the winner is ={winner}")
                if winner == 0:
                    agentRL += 1
                elif winner == 1:
                    agentRandom += 1
                else:
                    draw += 1
                # or winner == -1
                if winner == 0 :
                    self.learn(self.__players[0])
                else:
                    self.__players[0].historyOfMoves = []
                    self.__players[0].historyOfPiece = []


            print(f"RL wins: {agentRL} and Random wins: {agentRandom} and Draw is: {draw}, ")
            return self.__players[0].placeWeightDict, self.__players[0].pieceWeightDict

    def learn(self, player):
        target = 0
        for prev, reward in reversed(player.historyOfMoves):
            player.placeWeightDict[prev] = player.placeWeightDict[prev] + player.alpha * (target - player.placeWeightDict[prev])
            target += reward

        target = 0
        for prev, reward in reversed(player.historyOfPiece):
            player.pieceWeightDict[prev] = player.pieceWeightDict[prev] + player.alpha * (target - player.pieceWeightDict[prev])
            target += reward

        player.historyOfMoves= []
        player.historyOfPiece = []

        player.randomFactor -= 10e-5  # decrease random factor each episode of play

    def calcPlaceWeight(self, pieceIndex, placeXY):
        grandPrise = 10e10
        worstPrise = -grandPrise
        # 2 file 5
        thresholdFor3 = 2
        copyOfBoard = self.get_board_status()
        self.updatePlayedPiecePlace()
        thisFreePlacesRewardForGivenPiece = []

        x, y = placeXY
        self.__board = copy.deepcopy(copyOfBoard)
        self.__board[x, y] = pieceIndex
        _, dictH = self.__check_horizontal()
        _, dictV = self.__check_vertical()
        _, dictD = self.new_check_diagonal()
        dictLTR, dictRTL = dictD["LTR"], dictD["RTL"]
        if dictH is not None and dictV is not None and dictLTR is not None and dictRTL is not None:
            propertyH = dictH[x]
            propertyV = dictV[y]
            thisStepH = numpy.array(
                [propertyH.high, propertyH.notHigh, propertyH.colored, propertyH.notColored, propertyH.solid,
                 propertyH.notSolid, propertyH.square,
                 propertyH.notSquare])
            thisStepV = numpy.array(
                [propertyV.high, propertyV.notHigh, propertyV.colored, propertyV.notColored, propertyV.solid,
                 propertyV.notSolid, propertyV.square,
                 propertyV.notSquare])
            l1 = dictLTR[0]
            l2 = dictLTR[1]
            l3 = dictLTR[2]
            l4 = dictLTR[3]
            thisStepL = numpy.array(
                [l1.high + l2.high + l3.high + l4.high, l1.notHigh + l2.notHigh + l3.notHigh + l4.notHigh,
                 l1.colored + l2.colored + l3.colored + l4.colored,
                 l1.notColored + l2.notColored + l3.notColored + l4.notColored,
                 l1.solid + l2.solid + l3.solid + l4.solid, l1.notSolid + l2.notSolid + l3.notSolid + l4.notSolid,
                 l1.square + l2.square + l3.square + l4.square,
                 l1.notSquare + l2.notSquare + l3.notSquare + l4.notSquare])

            r1 = dictRTL[0]
            r2 = dictRTL[1]
            r3 = dictRTL[2]
            r4 = dictRTL[3]
            thisStepR = numpy.array(
                [r1.high + r2.high + r3.high + r4.high, r1.notHigh + r2.notHigh + r3.notHigh + r4.notHigh,
                 r1.colored + r2.colored + r3.colored + r4.colored,
                 r1.notColored + r2.notColored + r3.notColored + r4.notColored,
                 r1.solid + r2.solid + r3.solid + r4.solid, r1.notSolid + r2.notSolid + r3.notSolid + r4.notSolid,
                 r1.square + r2.square + r3.square + r4.square,
                 r1.notSquare + r2.notSquare + r3.notSquare + r4.notSquare])

            h4 = len(thisStepH[thisStepH == 4])
            h3 = len(thisStepH[thisStepH == 3])
            h2 = len(thisStepH[thisStepH == 2])
            h1 = len(thisStepH[thisStepH == 1])
            h0 = len(thisStepH[thisStepH == 0])

            v4 = len(thisStepV[thisStepV == 4])
            v3 = len(thisStepV[thisStepV == 3])
            v2 = len(thisStepV[thisStepV == 2])
            v1 = len(thisStepV[thisStepV == 1])
            v0 = len(thisStepV[thisStepV == 0])

            l4 = len(thisStepL[thisStepL == 4])
            l3 = len(thisStepL[thisStepL == 3])
            l2 = len(thisStepL[thisStepL == 2])
            l1 = len(thisStepL[thisStepL == 1])
            l0 = len(thisStepL[thisStepL == 0])

            r4 = len(thisStepR[thisStepR == 4])
            r3 = len(thisStepR[thisStepR == 3])
            r2 = len(thisStepR[thisStepR == 2])
            r1 = len(thisStepR[thisStepR == 1])
            r0 = len(thisStepR[thisStepR == 0])

            temp = [h4, h3, h2, h1, h0, v4, v3, v2, v1, v0, l4, l3, l2, l1, l0, r4, r3, r2, r1, r0]
            forNormalize = numpy.array(temp)
            forNormalize = forNormalize[forNormalize != 0]
            totalValued = sum(forNormalize)

            if r4 != 0 or l4 != 0 or v4 != 0 or h4 != 0:
                thisFreePlacesRewardForGivenPiece.append(((x, y), grandPrise))
            else:
                if h3 + v3 + l3 + r3 > thresholdFor3:
                    thisFreePlacesRewardForGivenPiece.append(((x, y), worstPrise))
                else:
                    newPrise = -1 * (h3 + v3 + l3 + r3) + 2 * (h2 + v2 + l2 + r2) + 3 * (h1 + v1 + l1 + r1) + 5 * (
                            h0 + v0 + l0 + r0)
                    newPrise = newPrise / totalValued
                    thisFreePlacesRewardForGivenPiece.append(((x, y), newPrise))

        else:
            self.__board = copyOfBoard
            return grandPrise

        # sortedList = sorted(thisFreePlacesRewardForGivenPiece, key=cmp_to_key(self.compare), reverse=True)
        bestPlaceForGivenPiece, bestPriseForGivenPiece = thisFreePlacesRewardForGivenPiece[0]
        self.__board = copyOfBoard
        return bestPriseForGivenPiece

    def run(self) -> int:
        '''
        Run the game (with output for every move)
        '''
        winner = -1
        while winner < 0 and not self.check_finished():
            # self.print()
            piece_ok = False
            while not piece_ok:
                piece_ok = self.select(self.__players[self.__current_player].choose_piece())
            piece_ok = False
            self.__current_player = (self.__current_player + 1) % self.MAX_PLAYERS
            # self.print()
            while not piece_ok:
                x, y = self.__players[self.__current_player].place_piece()
                piece_ok = self.place(x, y)
            winner = self.check_winner()
        # self.print()
        return winner