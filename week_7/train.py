from utils import *


if __name__ == "__main__" :
    # load data
    df = load_data()

    # prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)

    # transform data 
    data_transformer = transform_data
    X_train, X_test = data_transformer(X_train), data_transformer(X_test)

    # train model 
    dv, model = train(X_train, y_train)

    # evaluate model 
    evaluate(model, dv, X_test, y_test)

    # save model 
    save_model(dv, transform_data, model)


