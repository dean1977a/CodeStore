class GRU_V0a():
    def __init__(self, **kw):
        super(GRU_V0a, self).__init__(**kw)
    self.categorical=['app', 'device', 'os', 'channel', 'hour']
    self.continous=[col for col in features if col not in self.categorical]
        self.categorical_num = {
            'app': (769, 16),
            'device': (4228, 16),
            'os': (957, 16),
            'channel': (501, 8),
            'hour': (24, 8),
        }
    def build_model(self):
        categorial_inp = Input(shape=(len(self.categorical),))
        cat_embeds = []
        for idx, col in enumerate(self.categorical):
            x = Lambda(lambda x: x[:, idx,None])(categorial_inp)
            x = Embedding(self.categorical_num[col][0], self.categorical_num[col][1],input_length=1)(x)
            cat_embeds.append(x)
        embeds = concatenate(cat_embeds, axis=2)
        embeds = GaussianDropout(0.2)(embeds)
        continous_inp = Input(shape=(len(self.continous),))
        cx = Reshape([1,len(self.continous)])(continous_inp)
        x = concatenate([embeds, cx], axis=2)
        x = CuDNNGRU(128)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.20)(x)
        x = Dense(64)(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.20)(x)
        x = Dense(32)(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.05)(x)
        outp = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=[categorial_inp, continous_inp], output=outp)
        print(model.summary())
        return model
