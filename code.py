#15.12 - исправлена матрица ошибок, дерево принятия решений, обработка данных неправильного формата (xls, txt и др)
from tkinter import *
from tkinter import filedialog, messagebox
import tkinter.ttk as ttk

# Машинное обучение
import pandas as pd
from sklearn import ensemble, tree
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, plot_roc_curve

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# Матрицы ошибок
ConfMatrix = np.array([[0, 0], [0, 0]])

# Классификатор
clf = LogisticRegression()

# Названия переменных (для построения дерева)
labels = list()

# Результат прогноза для заемщиков
results = pd.DataFrame()

# Тестовые выборки (для ROC-кривой)
X_test = pd.DataFrame()
y_test = pd.Series()

# ФУНКЦИИ
# Чтение данных
def clicked_btn_readDataAppl():
    global pathAppl
    file = filedialog.askopenfilename()
    pathAppl.set(file)


def clicked_btn_readDataCred():
    global pathCred
    file = filedialog.askopenfilename()
    pathCred.set(file)


# Команды меню
def aboutProgram():

    newWindow = Toplevel(root)
    newWindow.title("О программе")
    newWindow.geometry('600x250')
    newWindow.resizable(width=False, height=False)

    Label(newWindow, text="Полное наименование:", font=("Arial Bold", 10, 'bold')).place(x=20, y=20)
    Label(newWindow, text="Автоматизированная система поддержки принятия\n"
                          "решений \"Кредитный скоринг\"", justify=LEFT,
          font=("Arial Bold", 10)).place(x=250, y=20)

    Label(newWindow, text="Сокращенное наименование:", font=("Arial Bold", 10, 'bold')).place(x=20, y=70)
    Label(newWindow, text="Кредитный скоринг", justify=LEFT, font=("Arial Bold", 10)).place(x=250, y=70)

    Label(newWindow, text="Разработчик:", font=("Arial Bold", 10, 'bold')).place(x=20, y=100)
    Label(newWindow, text="Чаганова Ольга Борисовна, 17ПМ(ба)ПММ", justify=LEFT, font=("Arial Bold", 10)).place(x=250, y=100)

    Label(newWindow, text="Руководители:", font=("Arial Bold", 10, 'bold')).place(x=20, y=130)
    Label(newWindow, text="Яркова Ольга Николаевна, к.э.н., доцент,\n"
                          "доцент кафедры математических методов\n"
                          "и моделей в экономике, ОГУ", justify=LEFT, font=("Arial Bold", 10)).place(x=250, y=130)

    Label(newWindow, text="Год:", font=("Arial Bold", 10, 'bold')).place(x=20, y=200)
    Label(newWindow, text="2020", justify=LEFT, font=("Arial Bold", 10)).place(x=250, y=200)


def helpMethodsAndMetrics():
    newWindow = Toplevel(root)
    newWindow.title("Справка: формат входных данных")
    newWindow.geometry('500x400')
    newWindow.resizable(width=False, height=False)

    label = Label(newWindow, text="Методы и метрики",
                  font=("Arial Bold", 12, 'bold'))
    label.place(x=180, y=10)

    text = Text(newWindow, width=55, height=20, wrap=WORD)
    text.insert(1.0, "Кредитный скоринг — это процесс оценки заемщика кредитной организацией."
                     " Скоринг производится с помощью построения скоринговой модели,"
                     " которая учитывает характеристики потенциального заемщика,"
                     " влияющие на его способность расплачиваться по кредиту.\n\n"
                     "В настоящей программе модель кредитного скоринга позволяет классифицировать заемщиков"
                     " на два класса: надежные и ненадежные. Для построения модели скоринга используются"
                     " следующие методы классификации:\n\n"
                     "  1. Алгоритм случайного леса;\n\n"
                     "  2. Логистическая регрессия;\n\n"
                     "  3. Модель искусственной нейронной сети.\n\n"
                     "Подробное описание моделей приведено в соответствующих пунктах меню.\n\n"
                     "Принцип работы программы состоит в следующем: данные о кредитной истории прошлых заемщиков"
                     "разбиваются на обучающую и тестовую выборки. По обучающей выборке происходит обучение модели"
                     " кредитного скоринга, по тестовой выборке рассчитываются показатели качества прогноза."
                     " Построенный классификатор используется для предсказания класса заемщиков, для которых требуется"
                     " оценить платежеспособность и данные о которых вводит пользователь.\n\n"
                     "Для того чтобы оценить качество полученного прогноза, рассчитывается матрица ошибок."
                     " Матрица ошибок представляет собой матрицу размерности 2*2, элементами которой являются:\n\n"
                     "                      [TN   FP]\n"
                     "                      [FN   TP]\n\n"
                     "где\n"
                     "  TP — истино-положительное решение;\n"
                     "  TN — истино-отрицательное решение;\n"
                     "  FP — ложно-положительное решение;\n"
                     "  FN — ложно-отрицательное решение.\n\n"
                     "На основе матрицы ошибок вычисляются следующие метрики:\n"
                     "  1. Accuracy - доля правильных ответов:\n\n"
                     "      Accuracy = (TP + TN) / (TP + FP + TN + FN)\n\n"
                     "  2. Precision - доля истинно положительных исходов из всего набора положительных меток:\n\n"
                     "              Precision = TP / (TP + FP)\n\n"
                     "  3. Recall -  доля истинно положительных исходов среди всех меток класса,"
                     " которые были определены как «положительный»:\n\n"
                     "              Recall = TP / (TP + FN)\n\n"
                     "  4. F-мера - гармоническое среднее между Precision и Recall:\n\n"
                     "    F-мера = 2*Precision*Recall / (Precision + Recall)\n\n"
                     "Для определения качества прогноза также может использовая ROC-кривая – график, показывающий"
                     " зависимость верно классифицируемых объектов положительного класса от ложно положительно"
                     " классифицируемых объектов негативного класса. Иначе, это соотношение True Positive Rate (Recall)"
                     " и False Positive Rate.\n\n"
                     "Количественной интерпретацие ROC-кривой является показатель AUC — площадь, ограниченная ROC-кривой"
                     " и осью доли ложных положительных классификаций. Чем выше показатель AUC, тем качественнее"
                     " классификатор, при этом значение 0,5 соответствует случайному гаданию. Значение менее 0,5"
                     " говорит, что классификатор действует с точностью до наоборот:"
                     " если положительные назвать отрицательными и наоборот, классификатор будет работать лучше. ")
    text.place(x=30, y=50)


def helpInputDataFormat():
    newWindow = Toplevel(root)
    newWindow.title("Справка: формат входных данных")
    newWindow.geometry('500x400')
    newWindow.resizable(width=False, height=False)

    label = Label(newWindow, text="Формат входных данных",
                  font=("Arial Bold", 12, 'bold'))
    label.place(x=140, y=10)

    text = Text(newWindow, width=55, height=20, wrap=WORD)
    text.insert(1.0, "Входные данные должны иметь расширение .csv (разделители - запятые).\n\n"
                     "Описания столбцов\n\n"
                     "1. Данные о кредитной истории прошлых заемщиков:\n\n"
                     "  ID: Индивидуальный идентификатор клиента.\n\n"
                     "  LIMIT_BAL: Сумма предоставленного кредита: включает как индивидуальный потребительский кредит, так и семейный (дополнительный) кредит.\n\n"
                     "  SEX: Пол (1 = мужской; 2 = женский).\n\n"
                     "  EDUCATION: Образование (1 = аспирантура; 2 = университет; 3 = средняя школа; 0, 4, 5, 6 = другие).\n\n"
                     "  MARRIAGE: Семейное положение (1 = женат; 2 = холост; 3 = развод; 0 = другие).\n\n"
                     "  AGE: Возраст (лет).\n\n"
                     "  PAY_1 - PAY_6: История прошлых платежей.\n"
                     "  PAY_1 = статус погашения в текущем месяце; PAY_2 = статус погашения в прошлом месяце;...\n"
                     "  Шкала измерения статуса погашения:\n"
                     "      -2: Нет потребления;\n"
                     "      -1: оплачено полностью;\n"
                     "       0: использование возобновляемого кредита;\n"
                     "       1 = отсрочка платежа на один месяц;\n"
                     "       2 = отсрочка платежа на два месяца;\n"
                     "       . . .\n"
                     "       8 = отсрочка платежа на восемь месяцев;\n"
                     "       9 = отсрочка платежа от девяти месяцев и более.\n\n"
                     "  BILL_AMT1 - BILL_AMT6: выписка о сумме счета.\n"
                     "  BILL_AMT1 = сумма выписки по счету за текущий месяц; BILL_AMT2 = сумма выписки по счету за прошлый месяц; ...\n\n"
                     "  PAY_AMT1 - PAY_AMT6: Сумма предыдущего платежа.\n"
                     "  PAY_AMT1 = сумма, выплаченная в текущем месяце.; PAY_AMT2 = сумма, выплаченная в прошлом месяце;...\n\n"
                     "  dpnm: поведение клиента:\n"
                     "  dpnm = 0, если кредит возвращен (надежный заемщик);\n"
                     "  dpnm = 1, если кредит не возвращен (ненадежный заемщик).\n"
                     "  Эти данные используются для обучения модели кредитного скоринга и тестирования ее адекватности.\n\n"
                     "2. Данные о заемщиках, для которых требуется составить прогноз кредитоспособности.\n"
                     "  Включают в себя все переменные, которые указаны выше, кроме переменной dpnm.")
    text.place(x=30, y=50)


def helpRandomForest():
    newWindow = Toplevel(root)
    newWindow.title("Справка: случайный лес")
    newWindow.geometry('500x400')
    newWindow.resizable(width=False, height=False)

    label = Label(newWindow, text="Алгоритм классификации Случайный лес (Random Forest)", font=("Arial Bold", 12, 'bold'))
    label.place(x=10, y=10)

    text = Text(newWindow, width=55, height=20, wrap=WORD)
    text.insert(1.0, "Случайный лес (Random Forest)\n\n"
                     "Случайный лес - это алгоритм классификации, заключающийся в использовании ансамбля решающих деревьев. "
                     "Каждое дерево решений – это способ представления правил в иерархической структуре. "
                     "Основа такой структуры – ответы «Да» или «Нет» на ряд вопросов.\n\n"
                     "Параметры модели:\n"
                     "1. Критерий расщепления:\n"
                     "  - gini - критерий расщепления Джини;\n"
                     "  - entropy - мера энтропии.\n\n"
                     "2. Максимальная глубина дерева - количество вершин в самом длинном пути от корня до самого дальнего листа. "
                     "Рекомендуемое значение - от 3 до 8, чтобы избежать переобучения модели.\n\n"
                     "3. Минимальное число объектов для разбиения - минимальное число объектов, при котором выполняется расщепление на классы.\n\n"
                     "4. Число деревьев - число деревьев в ансамбле. Чем больше деревьев, тем лучше качество модели, но время работы алгоритма также пропорционально увеличивается.\n\n"
                     "5. Число признаков для выбора расщепления - максимальное число признаков, по которым ищется лучшее разбиение в дереве.\n\n"
                     "6. Ограничение на число объектов в листьях - минимальное число объектов, находящихся в одном листе.")
    text.place(x=30, y=50)


def helpLogitRegression():
    newWindow = Toplevel(root)
    newWindow.title("Справка: логистическая регрессия")
    newWindow.geometry('500x400')
    newWindow.resizable(width=False, height=False)

    label = Label(newWindow, text="Алгоритм классификации Логистическая регрессия\n (Logistic Regression)",
                  font=("Arial Bold", 12, 'bold'))
    label.place(x=20, y=10)

    text = Text(newWindow, width=55, height=18, wrap=WORD)
    text.insert(1.0, "Логистическая регрессия (Logistic Regression)\n\n"
                     "Логистическая регрессия - метод построения линейного классификатора, позволяющий оценивать апостериорные вероятности принадлежности объектов классам"
                     "и на их основе классифицировать объекты.\n\n"
                     "Параметры модели:\n"
                     "1. Алгоритм оптимизации:\n"
                     "  - newton-cg - алгоритм сопряженных градиентов;\n"
                     "  - lbfgs - Алгоритм Бройдена — Флетчера — Гольдфарба — Шанно;\n"
                     "  - liblinear - хорошо подходит для маленьких датасетов;\n"
                     "  - sag - стохастический средний градиент: на больших данных работает быстрее, чем liblinear;\n"
                     "  - saga - на больших данных работает быстрее, чем liblinear.\n\n"
                     "2. Вид регуляризации:\n"
                     "  - l1 - регуляризация через манхэттенское расстояние;\n"
                     "  - l2 - регуляризация Тихонова;\n"
                     "  - elasticnet - Упругая сетевая регуляризация;\n"
                     "  - none - регуляризация не применяется.\n\n"
                     "3. Параметр регуляризации - инверсия силы регуляризации. Меньшие значения указывают на более сильную регуляризацию.\n\n"
                     "4. Максимальное число итераций - максимальное количество итераций, необходимых для схождения алгоритма.\n")
    text.place(x=30, y=70)


def helpNeuralNetwork():
    newWindow = Toplevel(root)
    newWindow.title("Справка: нейронная сеть")
    newWindow.geometry('500x400')
    newWindow.resizable(width=False, height=False)

    label = Label(newWindow, text="Классификация с помощью нейронных сетей\n(Neural Network Classification)", font=("Arial Bold", 12, 'bold'))
    label.place(x=60, y=10)

    text = Text(newWindow, width=55, height=18, wrap=WORD)
    text.insert(1.0, "Нейронная сеть (Neural Network)\n\n"
                     "Сети с прямой связью являются универсальным средством аппроксимации функций, что позволяет их использовать в решении задач классификации. "
                     "Как правило, нейронные сети оказываются наиболее эффективным способом классификации, потому что генерируют фактически большое число регрессионных моделей.\n\n"
                     "Параметры модели:\n"
                     "1. Функциия активации скрытого слоя:\n"
                     "  - identity - линейная функция активации: f(x) = x;\n"
                     "  - logistic - сигмоида: f(x) = 1 / (1 + exp(-x)).\n"
                     "  - tanh - гиперболический тангенс: f(x) = tanh(x)\n"
                     "  - relu - выпрямленная линейная функция: f(x) = max(0,x).\n\n "
                     "2. Алгоритм оптимизации:\n"
                     "  - lbfgs - квазиньютоновский алгоритм оптимизации;\n"
                     "  - sgd - стохастический градиентный спуск;\n"
                     "  - adam - оптимизатор на основе стохастического градиента.\n"
                     "  Adam хорошо работает с относительно большими наборами данных (несколько тысяч записей). Однако для небольших наборов данных lbfgs может сходиться быстрее и работать лучше.\n\n"
                     "3. Коэффициент скорости обучения:\n"
                     "  - constant - константа;\n"
                     "  - invscaling - постепенно снижает скорость обучения на каждом временном шаге, используя обратное масштабирование;\n"
                     "  - adaptive - поддерживает постоянную скорость обучения на уровне Learning_rate_init, пока потери в обучении продолжают уменьшаться."
                     " Каждый раз, когда две последовательные эпохи не могут уменьшить потери в обучении или не могут увеличить оценку валидации, текущая скорость обучения делится на 5.\n\n"
                     "4. Максимальное число итераций - максимальное количество итераций, необходимых для схождения алгоритма.\n")

    text.place(x=30, y=70)


# Изменение фокуса настраиваемых полей
def changeStateRF():
    #global state
    if cvar1.get():
        combo_RFCriterion.config(state='normal')
        e_RFMaxDepth.config(state='normal')
        e_RFMinSamplesSplit.config(state='normal')
        e_RFNEstimators.config(state='normal')
        e_RFMaxFeatures.config(state='normal')
        e_RFMinSamplesLeaf.config(state='normal')

    else:
        combo_RFCriterion.config(state='disabled')
        e_RFMaxDepth.config(state='disabled')
        e_RFMinSamplesSplit.config(state='disabled')
        e_RFNEstimators.config(state='disabled')
        e_RFMaxFeatures.config(state='disabled')
        e_RFMinSamplesLeaf.config(state='disabled')


def changeStateLR():
    if cvar2.get():
        combo_LRSolver.config(state='normal')
        combo_LRPenalty.config(state='normal')
        e_LRC.config(state='normal')
        e_LRMaxIter.config(state='normal')

    else:
        combo_LRSolver.config(state='disabled')
        combo_LRPenalty.config(state='disabled')
        e_LRC.config(state='disabled')
        e_LRMaxIter.config(state='disabled')


def changeStateNN():
    if cvar3.get():
        combo_NNActivation.config(state='normal')
        e_NNNumOfHiddenLayers.config(state='normal')
        e_NNNumOfNeurons.config(state='normal')
        combo_NNSolver.config(state='normal')
        combo_NNLearningRate.config(state='normal')
        e_NNLearningRateInit.config(state='normal')
        e_NNMaxIter.config(state='normal')

    else:
        combo_NNActivation.config(state='disabled')
        e_NNNumOfHiddenLayers.config(state='disabled')
        e_NNNumOfNeurons.config(state='disabled')
        combo_NNSolver.config(state='disabled')
        combo_NNLearningRate.config(state='disabled')
        e_NNLearningRateInit.config(state='disabled')
        e_NNMaxIter.config(state='disabled')


# Классификация: Случайный лес
def classifyRandomForest():

    if cvar1.get():
        criterion = combo_RFCriterion.get()
        try:
            min_samples_split = int(e_RFMinSamplesSplit.get())
            max_depth = int(e_RFMaxDepth.get())
            n_estimators = int(e_RFNEstimators.get())
            max_features = int(e_RFMaxFeatures.get())
            min_samples_leaf = int(e_RFMinSamplesLeaf.get())
            if (min_samples_split < 1) or (max_depth < 1) or (n_estimators < 1) or (max_features < 1) or (min_samples_leaf < 1):
                raise Exception()
        except ValueError:
            messagebox.showerror(title="Ошибка ввода",
                                 message="Входные данные имеют неправильный тип.\nЗадайте параметры заново.")
        except Exception:
            messagebox.showerror(title="Ошибка ввода",
                                 message="Значения параметров должны быть больше единицы.\nЗадайте параметры заново.")

    else:
        criterion = 'gini'
        min_samples_split = 23
        max_depth = 3
        n_estimators = 10
        max_features = "auto"
        min_samples_leaf = 1


    try:
        pathCred = e_pathCred.get()
        credit_history = pd.read_csv(pathCred)

        pathAppl = e_pathAppl.get()
        applicants = pd.read_csv(pathAppl)

        if (credit_history.shape[0] < 50 or applicants.shape[0] < 1): raise Exception

        credit_history = credit_history.drop(['ID'], axis=1)
        credit_history = credit_history.drop_duplicates()

        X = credit_history[credit_history.columns[:-1]]  # объясняющие переменные (все столбцы за исключением последнего)
        y = credit_history['dpnm']  # последний столбец

        global labels
        labels = list(X)

        global X_test, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    except pd.errors.ParserError:
        messagebox.showerror(title="Некорректный тип данных",
                             message="Исходные данные должны иметь расширение .csv.")

    except Exception:
        messagebox.showerror(title="Некорректный формат данных",
                             message="Исходные имеют неверный формат.\nПроверьте правильность записи данных"
                                     " (пункт меню \"Формат входных данных\") и убедитесь, что число наблюдений"
                                     " для кредитной истории больше пятидесяти, а для новых заемщиков - не меньше одного."
                                     " Повторите загрузку данных.")
    global clf
    forest = ensemble.RandomForestClassifier(criterion=criterion, min_samples_split=min_samples_split,
                                             max_depth=max_depth, n_estimators=n_estimators, max_features=max_features,
                                             min_samples_leaf=min_samples_leaf, random_state=42)
    clf = forest

    forest.fit(X_train, y_train)

    y_pred = forest.predict(X_test) # предсказание по тестовой выборке для расчета качества прогноза

    # Метрики
    global RF_Accuracy, RF_Precision, RF_Recall, RF_F
    RF_Accuracy.set(round(forest.score(X_test, y_test), 5))
    RF_Precision.set(round(precision_score(y_test, y_pred), 5))
    RF_Recall.set(round(recall_score(y_test, y_pred), 5))
    RF_F.set(round(f1_score(y_test, y_pred), 5))

    # Кнопки
    btn_RFMatrix.config(state="normal")
    btn_RFDecisionTree.config(state="normal")
    btn_RFRocCurve.config(state="normal")
    btn_RFSaveResults.config(state="normal")

    # Матрица ошибок
    global ConfMatrix
    ConfMatrix = confusion_matrix(y_test, y_pred)

    try:
        # Предсказание для заявителей
        ID = applicants['ID']
        applicants = applicants.drop(['ID'], axis=1)
        applicants = applicants.assign(Class=forest.predict(applicants))
        applicants = applicants.assign(ID=ID)

        global results
        results = applicants.loc[:, ['ID', 'Class']]

    except Exception:
        messagebox.showerror(title="Ошибка ввода данных", message="Данные о новых заемщиках имеют неверный формат. "
                                                                  "Классификация не будет осуществлена. "
                                                                  "Повторите загрузку данных.")
        btn_RFSaveResults.config(state='disabled')
        btn_LRSaveResults.config(state='disabled')
        btn_NNSaveResults.config(state='disabled')


# Классификация: Логистическая регрессия
def classifyLogisticRegression():
    if cvar2.get():
        solver = combo_LRSolver.get()
        penalty = combo_LRPenalty.get()
        try:
            C = int(e_LRC.get())
            max_iter = int(e_LRMaxIter.get())
            if (C < 1) or (max_iter < 1):
                raise Exception()
        except ValueError:
            messagebox.showerror(title="Ошибка ввода",
                                 message="Входные данные имеют неправильный тип.\nЗадайте параметры заново.")
        except Exception:
            messagebox.showerror(title="Ошибка ввода",
                                 message="Значения параметров должны быть больше единицы.\nЗадайте параметры заново.")

    else:
        solver = 'liblinear'
        penalty = 'l2'
        C = 1
        max_iter = 100

    pathCred = e_pathCred2.get()
    credit_history = pd.read_csv(pathCred)

    pathAppl = e_pathAppl2.get()
    applicants = pd.read_csv(pathAppl)

    try:
        credit_history = credit_history.drop(['ID'], axis=1)
        credit_history = credit_history.drop_duplicates()

        X = credit_history[credit_history.columns[:-1]]  # объясняющие переменные (все столбцы за исключением последнего)
        y = credit_history['dpnm']  # последний столбец

        global X_test, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    except Exception:
        messagebox.showerror(title="Ошибка ввода данных", message="Данные о кредитной истории имеют неверный формат."
                                                                  " Повторите загрузку данных.")

    global clf
    logit = LogisticRegression(solver=solver, penalty=penalty, C=C, max_iter=max_iter, random_state=42)
    clf = logit

    logit.fit(X_train, y_train)

    y_pred = logit.predict(X_test)  # предсказание по тестовой выборке для расчета качества прогноза

    # Метрики
    global LR_Accuracy, LR_Precision, LR_Recall, LR_F
    LR_Accuracy.set(round(logit.score(X_test, y_test), 5))
    LR_Precision.set(round(precision_score(y_test, y_pred), 5))
    LR_Recall.set(round(recall_score(y_test, y_pred), 5))
    LR_F.set(round(f1_score(y_test, y_pred), 5))

    # Кнопки
    btn_LRMatrix.config(state="normal")
    btn_LRRocCurve.config(state="normal")
    btn_LRSaveResults.config(state="normal")

    # Матрица ошибок
    global ConfMatrix
    ConfMatrix = confusion_matrix(y_test, y_pred)

    try:
    # Предсказание для заявителей
        ID = applicants['ID']
        applicants = applicants.drop(['ID'], axis=1)
        applicants = applicants.assign(Class=logit.predict(applicants))
        applicants = applicants.assign(ID=ID)

        global results
        results = applicants.loc[:, ['ID', 'Class']]

    except Exception:
        messagebox.showerror(title="Ошибка ввода данных", message="Данные о заемщиках имеют неверный формат. "
                                                                  "Классификация не будет осуществлена. "
                                                                  "Повторите загрузку данных.")
        btn_RFSaveResults.config(state='disabled')
        btn_LRSaveResults.config(state='disabled')
        btn_NNSaveResults.config(state='disabled')

# Классификация: Нейронная сеть
def classifyNeuralNetwork():
    if cvar3.get():
        activation = combo_NNActivation.get()
        solver = combo_NNSolver.get()
        learning_rate = combo_NNLearningRate.get()
        try:
            numOfHiddenLayers = int(e_NNNumOfHiddenLayers.get())
            numOfNeurons = int(e_NNNumOfNeurons.get())
            learning_rate_init = float(e_NNLearningRateInit.get())
            max_iter = int(e_NNMaxIter.get())

            if (numOfHiddenLayers < 1) or (numOfNeurons < 1) or (max_iter < 1) or (learning_rate_init < -1):
                raise Exception()
        except ValueError:
            messagebox.showerror(title="Ошибка ввода",
                                 message="Входные данные имеют неправильный тип.\nЗадайте параметры заново.")
        except Exception:
            messagebox.showerror(title="Ошибка ввода (нейронная сеть)",
                                 message="Значения параметров должны быть больше единицы.\nЗадайте параметры заново.")

        hidden_layer_sizes = (numOfNeurons,)
        for i in range(numOfHiddenLayers - 1):
            hidden_layer_sizes += (numOfNeurons,)
            
    else:
        activation = 'relu'
        solver = 'adam'
        learning_rate = 'constant'
        learning_rate_init = 0.001
        max_iter = 200
        hidden_layer_sizes = (100,)

    try:
        pathCred = e_pathCred3.get()
        credit_history = pd.read_csv(pathCred)

        pathAppl = e_pathAppl3.get()
        applicants = pd.read_csv(pathAppl)

        credit_history = credit_history.drop(['ID'], axis=1)
        credit_history = credit_history.drop_duplicates()

        X = credit_history[credit_history.columns[:-1]]  # объясняющие переменные (все столбцы за исключением последнего)
        y = credit_history['dpnm']  # последний столбец

        global X_test, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    except Exception:
        messagebox.showerror(title="Ошибка ввода данных", message="Данные о кредитной истории имеют неверный формат."
                                                                      " Повторите загрузку данных.")

    global clf
    neuralnet = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                              learning_rate=learning_rate, learning_rate_init=learning_rate_init, max_iter=max_iter,
                              random_state=42)
    clf = neuralnet

    neuralnet.fit(X_train, y_train)
    y_pred = neuralnet.predict(X_test)  # предсказание по тестовой выборке для расчета качества прогноза

    # Метрики
    global NN_Accuracy, NN_Precision, NN_Recall, NN_F
    NN_Accuracy.set(round(neuralnet.score(X_test, y_test), 5))
    NN_Precision.set(round(precision_score(y_test, y_pred), 5))
    NN_Recall.set(round(recall_score(y_test, y_pred), 5))
    NN_F.set(round(f1_score(y_test, y_pred), 5))

    # Кнопки
    btn_NNMatrix.config(state="normal")
    btn_NNRocCurve.config(state="normal")
    btn_NNSaveResults.config(state="normal")

    # Матрица ошибок
    global ConfMatrix
    ConfMatrix = confusion_matrix(y_test, y_pred)

    try:
        # Предсказание для заявителей
        ID = applicants['ID']
        applicants = applicants.drop(['ID'], axis=1)
        applicants = applicants.assign(Class=neuralnet.predict(applicants))
        applicants = applicants.assign(ID=ID)

        global results
        results = applicants.loc[:, ['ID', 'Class']]

    except Exception:
        messagebox.showerror(title="Ошибка ввода данных", message="Данные о заемщиках имеют неверный формат. "
                                                                  "Классификация не будет осуществлена. "
                                                                  "Повторите загрузку данных.")
        btn_RFSaveResults.config(state='disabled')
        btn_LRSaveResults.config(state='disabled')
        btn_NNSaveResults.config(state='disabled')


# Вывод матрицы ошибок
def ShowConfusionMatrix():
    fig = plt.figure()
    ax = plt.subplot()
    sns.heatmap(ConfMatrix, annot=True, cbar=False, cmap='twilight', linewidth=0.5, fmt="d")
    plt.title('Матрица ошибок при классификации заемщиков')

    plt.xlabel('Предсказанные значения классов заемщиков')
    plt.ylabel('Истинные значения')
    ax.xaxis.set_ticklabels(['Надежный', 'Ненадежный'])
    ax.yaxis.set_ticklabels(['Надежный', 'Ненадежный'])
    plt.show()


# Визуализация дерева
def VisualizeDecisionTree():
    estimator = clf.estimators_[0]
    fig = plt.figure(figsize=(21, 6))
    fig = tree.plot_tree(estimator, fontsize=6, feature_names=labels, filled=True,
                         class_names=["Надежный", "Ненадежный"], proportion=True)
    plt.show()


# Построение ROC-кривой
def BuiltRocCurve():
    plot_roc_curve(clf, X_test, y_test)
    plt.show()


# Сохранение результатов прогнозирования
def SaveResults():
    export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')
    results.to_csv(export_file_path, index=True, header=True)


# Основная форма
root = Tk()
root.title('Кредитный скоринг')
root.geometry('600x600')
root.resizable(width=False, height=False)

# Меню
mainmenu = Menu(root)
root.config(menu=mainmenu)

helpMenu = Menu(mainmenu, tearoff=0)
helpMenu.add_command(label="Методы и метрики", command=helpMethodsAndMetrics)
helpMenu.add_command(label="Формат входных данных", command=helpInputDataFormat)
helpMenu.add_command(label="Случайный лес", command=helpRandomForest)
helpMenu.add_command(label="Логистическая регрессия", command=helpLogitRegression)
helpMenu.add_command(label="Нейронная сеть", command=helpNeuralNetwork)

mainmenu.add_cascade(label='О программе', command=aboutProgram)
mainmenu.add_cascade(label='Справка', menu=helpMenu)

# Вкладки
note = ttk.Notebook(root)
note.pack(fill='both', expand='yes')

tab1 = Frame(note)
tab2 = Frame(note)
tab3 = Frame(note)

note.add(tab1, text='Случайный лес')
note.add(tab2, text='Логистическая регрессия')
note.add(tab3, text='Нейронная сеть')

# --------------------Вкладка "СЛУЧАЙНЫЙ ЛЕС"--------------------------------

# Кредитная история
Label(tab1, text="Данные о кредитной истории", font=("Arial Bold", 12)).place(x=30, y=30)

btn_readDataCred = Button(tab1, text="Загрузить", font=("Arial Bold", 9), command=clicked_btn_readDataCred, width=10)
btn_readDataCred.place(x=275, y=30)

pathCred = StringVar()

e_pathCred = Entry(tab1, textvariable=pathCred, width=28, state='disabled')
e_pathCred.place(x=400, y=30)

# Заемщики
Label(tab1, text="Данные о заемщиках", font=("Arial Bold", 12)).place(x=30, y=70)

btn_readDataAppl = Button(tab1, text="Загрузить", font=("Arial Bold", 9), command=clicked_btn_readDataAppl, width=10)
btn_readDataAppl.place(x=275, y=70)

pathAppl = StringVar()

e_pathAppl = Entry(tab1, textvariable=pathAppl, width=28, state='disabled')
e_pathAppl.place(x=400, y=70)


# Фокус
cvar1 = BooleanVar()
cvar1.set(0)
chb1 = Checkbutton(tab1, text='Настраиваемые параметры', variable=cvar1, onvalue=1, offvalue=0, font=("Arial Bold", 12, 'bold'), command=changeStateRF)
chb1.deselect()

# Настраиваемые параметры

# Рамка
lF1 = LabelFrame(tab1, labelwidget=chb1, font=("Arial Bold", 12, 'bold'), relief="groove", width=540, height=170)

# Фон под рамку
c1 = Canvas(lF1, width=520, height=150, bg="pink")
c1.pack(side=LEFT)

myscrollbar = Scrollbar(lF1, orient="vertical", command=c1.yview)
myscrollbar.pack(side="right", fill="y")

c1.configure(yscrollcommand=myscrollbar.set)

c1.bind("<Configure>", lambda e: c1.configure(scrollregion=c1.bbox('all')))

frame1 = Frame(c1, width=520, height=250)
c1.create_window((0, 0), window=frame1, anchor='nw', width=520, height=250)
lF1.place(x=30, y=130)


# 1 Критерий расщепления
Label(frame1, text="Критерий расщепления", font=("Arial Bold", 12)).place(x=5, y=5)
combo_RFCriterion = ttk.Combobox(frame1, values=['entropy', 'gini'], width=17, state='disabled')
combo_RFCriterion.place(x=380, y=5)
combo_RFCriterion.current(0)

# 2 Максимальная глубина
Label(frame1, text="Максимальная глубина дерева", font=("Arial Bold", 12)).place(x=5, y=45)
v2 = StringVar()
v2.set("3")
e_RFMaxDepth = Entry(frame1, width=20, textvariable=v2, state='disabled')
e_RFMaxDepth.place(x=380, y=45)

# 3 Минимальное число объектов для разбиения
Label(frame1, text="Минимальное число объектов для разбиения", font=("Arial Bold", 12)).place(x=5, y=85)
v3 = StringVar()
v3.set("2")
e_RFMinSamplesSplit = Entry(frame1, width=20, textvariable=v3, state='disabled')
e_RFMinSamplesSplit.place(x=380, y=85)

# 4 Число деревьев
Label(frame1, text="Число деревьев", font=("Arial Bold", 12)).place(x=5, y=125)
v4 = StringVar()
v4.set("10")
e_RFNEstimators = Entry(frame1, width=20, textvariable=v4, state='disabled')
e_RFNEstimators.place(x=380, y=125)

# 5 Число признаков для выбора расщепления
Label(frame1, text="Число признаков для выбора расщепления", font=("Arial Bold", 12)).place(x=5, y=165)
v5 = StringVar()
v5.set("23")
e_RFMaxFeatures = Entry(frame1, width=20, textvariable=v5, state='disabled')
e_RFMaxFeatures.place(x=380, y=165)

# 6 Ограничение на число объектов в листьях
Label(frame1, text="Ограничение на число объектов в листьях", font=("Arial Bold", 12)).place(x=5, y=205)
v6 = StringVar()
v6.set("1")
e_RFMinSamplesLeaf = Entry(frame1,  width=20, textvariable=v6, state='disabled')
e_RFMinSamplesLeaf.place(x=380, y=205)


# Кнопка Составить прогноз
btn_forecast = Button(tab1, text="Составить прогноз", font=("Arial Bold", 12), width=20, command=classifyRandomForest)
btn_forecast.place(x=220, y=340)

# Метрики прогноза
Label(tab1, text="Метрики прогноза", font=("Arial Bold", 12, 'bold')).place(x=30, y=390)

Label(tab1, text="Accuracy", font=("Arial Bold", 12)).place(x=30, y=430)
RF_Accuracy = StringVar()
e_RFAccuracy = Entry(tab1, width=20, textvariable=RF_Accuracy)
e_RFAccuracy.place(x=130, y=430)

Label(tab1, text="Precision", font=("Arial Bold", 12)).place(x=30, y=460)
RF_Precision = StringVar()
e_RFPrecision = Entry(tab1, width=20, textvariable=RF_Precision)
e_RFPrecision.place(x=130, y=460)

Label(tab1, text="Recall", font=("Arial Bold", 12)).place(x=30, y=490)
RF_Recall = StringVar()
e_RFRecall = Entry(tab1, width=20, textvariable=RF_Recall)
e_RFRecall.place(x=130, y=490)

Label(tab1, text="F-мера", font=("Arial Bold", 12)).place(x=30, y=520)
RF_F = StringVar()
e_RFF = Entry(tab1, width=20, textvariable=RF_F)
e_RFF.place(x=130, y=520)

# Кнопка Матрица ошибок
btn_RFMatrix = Button(tab1, text="Матрица ошибок", font=("Arial Bold", 10), width=20,
                      command=ShowConfusionMatrix, state='disabled')
btn_RFMatrix.place(x=400, y=400)

# Кнопка Построить дерево
btn_RFDecisionTree = Button(tab1, text="Построить дерево", font=("Arial Bold", 10), width=20,
                            command=VisualizeDecisionTree, state='disabled')
btn_RFDecisionTree.place(x=400, y=440)

# Кнопка Построить ROC-кривую
btn_RFRocCurve = Button(tab1, text="ROC-кривая", font=("Arial Bold", 10), width=20,
                            command=BuiltRocCurve, state='disabled')
btn_RFRocCurve.place(x=400, y=480)

# Кнопка Сохранить результат прогноза
btn_RFSaveResults = Button(tab1, text="Сохранить результат", font=("Arial Bold", 10), width=20,
                           command=SaveResults, state='disabled')
btn_RFSaveResults.place(x=400, y=520)


# -----------Вкладка "ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ"-----------------------------

# Кредитная история
Label(tab2, text="Данные о кредитной истории", font=("Arial Bold", 12)).place(x=30, y=30)

btn_readDataCred2 = Button(tab2, text="Загрузить", font=("Arial Bold", 9), command=clicked_btn_readDataCred, width=10)
btn_readDataCred2.place(x=275, y=30)

e_pathCred2 = Entry(tab2, textvariable=pathCred, width=28, state='disabled')
e_pathCred2.place(x=400, y=30)

# Заемщики
Label(tab2, text="Данные о заемщиках", font=("Arial Bold", 12)).place(x=30, y=70)

btn_readDataAppl2 = Button(tab2, text="Загрузить", font=("Arial Bold", 9), command=clicked_btn_readDataAppl, width=10)
btn_readDataAppl2.place(x=275, y=70)

e_pathAppl2 = Entry(tab2, textvariable=pathAppl, width=28, state='disabled')
e_pathAppl2.place(x=400, y=70)


# Фокус
cvar2 = BooleanVar()
cvar2.set(0)
chb2 = Checkbutton(tab2, text='Настраиваемые параметры', variable=cvar2, onvalue=1, offvalue=0, font=("Arial Bold", 12, 'bold'), command=changeStateLR)
chb2.deselect()

# Настраиваемые параметры

# Рамка
lF2 = LabelFrame(tab2, labelwidget=chb2, font=("Arial Bold", 12, 'bold'), relief="groove", width=540, height=170)

# Фон под рамку
c2 = Canvas(lF2, width=520, height=150, bg="pink")
c2.pack(side=LEFT)

myscrollbar2 = Scrollbar(lF2, orient="vertical", command=c2.yview)
myscrollbar2.pack(side="right", fill="y")

c2.configure(yscrollcommand=myscrollbar2.set)

c2.bind("<Configure>", lambda e: c2.configure(scrollregion=c2.bbox('all')))

frame2 = Frame(c2, width=520, height=150)
c2.create_window((0, 0), window=frame2, anchor='nw', width=520, height=150)
lF2.place(x=30, y=130)


# 1 Алгоритм оптимизации
Label(frame2, text="Алгоритм оптимизации", font=("Arial Bold", 12)).place(x=5, y=5)
combo_LRSolver = ttk.Combobox(frame2, values=['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'], width=17, state='disabled')
combo_LRSolver.place(x=380, y=5)
combo_LRSolver.current(0)

# 2 Вид регуляризации
Label(frame2, text="Вид регуляризации", font=("Arial Bold", 12)).place(x=5, y=45)
combo_LRPenalty = ttk.Combobox(frame2, values=['l1', 'l2', 'elasticnet', 'none'], width=17, state='disabled')
combo_LRPenalty.place(x=380, y=45)
combo_LRPenalty.current(1)

# 3 Параметр регуляризации
Label(frame2, text="Параметр регуляризации", font=("Arial Bold", 12)).place(x=5, y=85)
w3 = StringVar()
w3.set("1")
e_LRC = Entry(frame2, width=20, textvariable=w3, state='disabled')
e_LRC.place(x=380, y=85)

# 4 Максимальное число итераций
Label(frame2, text="Максимальное число итераций", font=("Arial Bold", 12)).place(x=5, y=125)
w4 = StringVar()
w4.set("100")
e_LRMaxIter = Entry(frame2, width=20, textvariable=w4, state='disabled')
e_LRMaxIter.place(x=380, y=125)


# Кнопка Составить прогноз
btn_forecast2 = Button(tab2, text="Составить прогноз", font=("Arial Bold", 12), width=20, command=classifyLogisticRegression)
btn_forecast2.place(x=220, y=340)

# Метрики прогноза
Label(tab2, text="Метрики прогноза", font=("Arial Bold", 12, 'bold')).place(x=30, y=390)

Label(tab2, text="Accuracy", font=("Arial Bold", 12)).place(x=30, y=430)
LR_Accuracy = StringVar()
e_LRAccuracy = Entry(tab2, width=20, textvariable=LR_Accuracy)
e_LRAccuracy.place(x=130, y=430)

Label(tab2, text="Precision", font=("Arial Bold", 12)).place(x=30, y=460)
LR_Precision = StringVar()
e_LRPrecision = Entry(tab2, width=20, textvariable=LR_Precision)
e_LRPrecision.place(x=130, y=460)

Label(tab2, text="Recall", font=("Arial Bold", 12)).place(x=30, y=490)
LR_Recall = StringVar()
e_LRRecall = Entry(tab2, width=20, textvariable=LR_Recall)
e_LRRecall.place(x=130, y=490)

Label(tab2, text="F-мера", font=("Arial Bold", 12)).place(x=30, y=520)
LR_F = StringVar()
e_LRF = Entry(tab2, width=20, textvariable=LR_F)
e_LRF.place(x=130, y=520)

# Кнопка Матрица ошибок
btn_LRMatrix = Button(tab2, text="Матрица ошибок", font=("Arial Bold", 10), width=20, command=ShowConfusionMatrix, state='disabled')
btn_LRMatrix.place(x=400, y=420)

# Кнопка Построить дерево
btn_LRRocCurve = Button(tab2, text="ROC-кривая", font=("Arial Bold", 10), width=20, command=BuiltRocCurve, state='disabled')
btn_LRRocCurve.place(x=400, y=465)

# Кнопка Сохранить результат прогноза
btn_LRSaveResults = Button(tab2, text="Сохранить результат", font=("Arial Bold", 10), width=20,
                           command=SaveResults, state='disabled')
btn_LRSaveResults.place(x=400, y=510)


# ---------------------------Вкладка "НЕЙРОННАЯ СЕТЬ"---------------------------------------

# Кредитная история
Label(tab3, text="Данные о кредитной истории", font=("Arial Bold", 12)).place(x=30, y=30)

btn_readDataCred3 = Button(tab3, text="Загрузить", font=("Arial Bold", 9), command=clicked_btn_readDataCred, width=10)
btn_readDataCred3.place(x=275, y=30)

e_pathCred3 = Entry(tab3, textvariable=pathCred, width=28, state='disabled')
e_pathCred3.place(x=400, y=30)

# Заемщики
Label(tab3, text="Данные о заемщиках", font=("Arial Bold", 12)).place(x=30, y=70)

btn_readDataAppl3 = Button(tab3, text="Загрузить", font=("Arial Bold", 9), command=clicked_btn_readDataAppl, width=10)
btn_readDataAppl3.place(x=275, y=70)


e_pathAppl3 = Entry(tab3, textvariable=pathAppl, width=28, state='disabled')
e_pathAppl3.place(x=400, y=70)


# Фокус
cvar3 = BooleanVar()
cvar3.set(0)
chb3 = Checkbutton(tab3, text='Настраиваемые параметры', variable=cvar3, onvalue=1, offvalue=0, font=("Arial Bold", 12, 'bold'), command=changeStateNN)
chb3.deselect()

# Настраиваемые параметры

# Рамка
lF3 = LabelFrame(tab3, labelwidget=chb3, font=("Arial Bold", 12, 'bold'), relief="groove", width=540, height=170)

# Фон под рамку
c3 = Canvas(lF3, width=520, height=150, bg="pink")
c3.pack(side=LEFT)

myscrollbar3 = Scrollbar(lF3, orient="vertical", command=c3.yview)
myscrollbar3.pack(side="right", fill="y")

c3.configure(yscrollcommand=myscrollbar3.set)

c3.bind("<Configure>", lambda e: c3.configure(scrollregion=c3.bbox('all')))

frame3 = Frame(c3, width=520, height=250)
c3.create_window((0, 0), window=frame3, anchor='nw', width=520, height=300)
lF3.place(x=30, y=130)


# 1 Функция активации скрытого слоя
Label(frame3, text="Функция активации скрытого слоя", font=("Arial Bold", 12)).place(x=5, y=5)
combo_NNActivation = ttk.Combobox(frame3, values=["identity", "logistic", "tanh", "relu"], width=17, state='disabled')
combo_NNActivation.place(x=380, y=5)
combo_NNActivation.current(3)

# 2 Число скрытых слоев
Label(frame3, text="Число скрытых слоев", font=("Arial Bold", 12)).place(x=5, y=45)
u2 = StringVar()
u2.set("1")
e_NNNumOfHiddenLayers = Entry(frame3, width=20, textvariable=u2, state='disabled')
e_NNNumOfHiddenLayers.place(x=380, y=45)

# 3 Число нейронов в скрытых слоях
Label(frame3, text="Число нейронов в каждом скрытом слое", font=("Arial Bold", 12)).place(x=5, y=85)
u3 = StringVar()
u3.set("100")
e_NNNumOfNeurons = Entry(frame3, width=20, textvariable=u3, state='disabled')
e_NNNumOfNeurons.place(x=380, y=85)

# 4 Алгоритм оптимизации
Label(frame3, text="Алгоритм оптимизации", font=("Arial Bold", 12)).place(x=5, y=125)
combo_NNSolver = ttk.Combobox(frame3, values=["lbfgs", "sgd", "adam"], width=17, state='disabled')
combo_NNSolver.place(x=380, y=125)
combo_NNSolver.current(2)

# 5 Коэффициент скорости обучения
Label(frame3, text="Коэффициент скорости обучения", font=("Arial Bold", 12)).place(x=5, y=165)
combo_NNLearningRate = ttk.Combobox(frame3, values=["constant", "invscaling", "adaptive"], width=17, state='disabled')
combo_NNLearningRate.place(x=380, y=165)
combo_NNLearningRate.current(0)

# 6 Начальное значение коэффициента скорости обучения
Label(frame3, text="Начальное значение коэффициента", font=("Arial Bold", 12)).place(x=5, y=205)
u6 = StringVar()
u6.set("0.001")
e_NNLearningRateInit = Entry(frame3, width=20, textvariable=u6, state='disabled')
e_NNLearningRateInit.place(x=380, y=205)

# 7 Максимальное число итераций
Label(frame3, text="Максимальное число итераций", font=("Arial Bold", 12)).place(x=5, y=245)
u7 = StringVar()
u7.set("200")
e_NNMaxIter = Entry(frame3, width=20, textvariable=u7, state='disabled')
e_NNMaxIter.place(x=380, y=245)


# Кнопка Составить прогноз
btn_forecast3 = Button(tab3, text="Составить прогноз", font=("Arial Bold", 12), width=20, command=classifyNeuralNetwork)
btn_forecast3.place(x=220, y=340)

# newWindow = Toplevel(tab3)
# newWindow.geometry('150x50')
# progress = ttk.Progressbar(newWindow, orient=HORIZONTAL, length=100, mode='indeterminate')
# progress.place(x=20, y=10)

# Метрики прогноза
Label(tab3, text="Метрики прогноза", font=("Arial Bold", 12, 'bold')).place(x=30, y=390)

Label(tab3, text="Accuracy", font=("Arial Bold", 12)).place(x=30, y=430)
NN_Accuracy = StringVar()
e_NNAccuracy = Entry(tab3, textvariable=NN_Accuracy, width=20)
e_NNAccuracy.place(x=130, y=430)

Label(tab3, text="Precision", font=("Arial Bold", 12)).place(x=30, y=460)
NN_Precision = StringVar()
e_NNPrecision = Entry(tab3, textvariable=NN_Precision, width=20)
e_NNPrecision.place(x=130, y=460)

Label(tab3, text="Recall", font=("Arial Bold", 12)).place(x=30, y=490)
NN_Recall = StringVar()
e_NNRecall = Entry(tab3, textvariable=NN_Recall, width=20)
e_NNRecall.place(x=130, y=490)

Label(tab3, text="F-мера", font=("Arial Bold", 12)).place(x=30, y=520)
NN_F = StringVar()
e_NNF = Entry(tab3, textvariable=NN_F, width=20)
e_NNF.place(x=130, y=520)

# Кнопка Матрица ошибок
btn_NNMatrix = Button(tab3, text="Матрица ошибок", font=("Arial Bold", 10), width=20, command=ShowConfusionMatrix, state='disabled')
btn_NNMatrix.place(x=400, y=420)

# Кнопка Построить ROC-кривую
btn_NNRocCurve = Button(tab3, text="ROC-кривая", font=("Arial Bold", 10), width=20, command=BuiltRocCurve, state='disabled')
btn_NNRocCurve.place(x=400, y=465)

# Кнопка Сохранить результат прогноза
btn_NNSaveResults = Button(tab3, text="Сохранить результат", font=("Arial Bold", 10), width=20, command=SaveResults, state='disabled')
btn_NNSaveResults .place(x=400, y=510)

root.mainloop()
