#!/usr/bin/env python3

# vabamorf -> ekilex
TAG_MAP = {
    "sg n": "SgN", 
    "sg g": "SgG", 
    "sg p": "SgP", 
    "adt": "SgAdt", 
    "sg ill": "SgIll", 
    "sg in": "SgIn", 
    "sg el": "SgEl", 
    "sg all": "SgAll", 
    "sg ad": "SgAd", 
    "sg abl": "SgAbl", 
    "sg tr": "SgTr", 
    "sg ter": "SgTer", 
    "sg es": "SgEs", 
    "sg ab": "SgAb", 
    "sg kom": "SgKom",
    "pl n": "PlN", 
    "pl g": "PlG", 
    "pl p": "PlP",  
    "pl ill": "PlIll", 
    "pl in": "PlIn", 
    "pl el": "PlEl", 
    "pl all": "PlAll", 
    "pl ad": "PlAd", 
    "pl abl": "PlAbl", 
    "pl tr": "PlTr", 
    "pl ter": "PlTer", 
    "pl es": "PlEs", 
    "pl ab": "PlAb", 
    "pl kom": "PlKom",
    "b": "IndPrSg3", #kindel kõneviis olevik 3. isik ainsus aktiiv jaatav kõne
    "d": "IndPrSg2", #kindel kõneviis olevik 2. isik ainsus aktiiv jaatav kõne
    "da": "Inf", #infinitiiv jaatav kõne
    "des": "Ger", #gerundium jaatav kõne
    "ge": "ge", #käskiv kõneviis olevik 2. isik mitmus aktiiv jaatav kõne
    "gem": "ImpPrPl1", #käskiv kõneviis olevik 1. isik mitmus aktiiv jaatav kõne
    "gu": "ImpPrPs", #käskiv kõneviis olevik 3. isik ainsus/mitmus aktiiv jaatav kõne
    "ks": "KndPrPs", #tingiv kõneviis olevik 1./2./3. isik ainsus/mitmus aktiiv jaatav kõne
    "ksid": "KndPrSg2", #tingiv kõneviis olevik 2./3. isik ainsus aktiiv jaatav kõne
    "ksime": "KndPrPl1", #tingiv kõneviis olevik 1. isik mitmus aktiiv jaatav kõne
    "ksin": "KndPrSg1", #tingiv kõneviis olevik 1. isik ainsus aktiiv jaatav kõne
    "ksite": "KndPrPl2", #tingiv kõneviis olevik 2. isik mitmus aktiiv jaatav kõne
    "ma": "Sup", #supiin aktiiv jaatav kõne sisseütlev
    "maks": "SupTr", #supiin aktiiv jaatav kõne saav
    "mas": "SupIn", #supiin aktiiv jaatav kõne seesütlev
    "mast": "SupEl", #supiin aktiiv jaatav kõne seestütlev
    "mata": "SupAb", #supiin aktiiv jaatav kõne ilmaütlev
    "me": "IndPrPl1", #kindel kõneviis olevik 1. isik mitmus aktiiv jaatav kõne
    "n": "IndPrSg1", #kindel kõneviis olevik 1. isik ainsus aktiiv jaatav kõne
    "neg": "Neg", #eitav kõne
    "neg da": "IndPrIpsN", #nt. "polda"
    "neg ge": "geNeg", #käskiv kõneviis olevik 2. isik mitmus aktiiv eitav kõne
    "neg gem": "ImpPrPl1Neg", #käskiv kõneviis olevik 1. isik mitmus aktiiv eitav kõne
    "neg gu": "ImpPrPsNeg|ImpPrIpsNeg", #käskiv kõneviis olevik 3. isik ainsus/mitmus aktiiv eitav kõne / käskiv kõneviis olevik passiiv eitav kõne
    "neg ks": "KndPrPsNeg", #tingiv kõneviis olevik 1./2./3. isik ainsus/mitmus aktiiv eitav kõne
    "neg me": "ImpPrPl1Neg", #käskiv kõneviis olevik 1. isik mitmus aktiiv eitav kõne
    "neg nud": "PtsPtPsNeg", #kindel kõneviis lihtminevik 1./2./3. isik ainsus/mitmus aktiiv eitav kõne
    "neg nuks": "KndPtPsNeg", #tingiv kõneviis minevik 1. isik mitmus aktiiv eitav kõne
    "neg o": "ImpPrSg2Neg|IndPrPsNeg", #käskiv kõneviis olevik 2. isik ainsus aktiiv eitav kõne / kindel kõneviis olevik 1./2./3. isik ainsus/mitmus aktiiv eitav kõne
    "neg vat": "KvtPrPsNeg", #kaudne kõneviis olevik 1./2./3. isik ainsus/mitmus aktiiv eitav kõne
    "neg tud": "PtsPtIpsNeg", #kesksõna minevik passiiv eitav kõne
    "nud": "PtsPtPs", #kesksõna minevik aktiiv jaatav kõne
    "nuks": "KndPtPs", #tingiv kõneviis minevik 1./2./3. isik ainsus/mitmus aktiiv jaatav kõne
    "nuksid": "KndPtSg2|KndPtPl3", #tingiv kõneviis minevik 2. isik ainsus aktiiv jaatav kõne / tingiv kõneviis minevik 3. isik mitmus aktiiv jaatav kõne
    "nuksime": "KndPtPl1", #tingiv kõneviis minevik 1. isik mitmus aktiiv jaatav kõne
    "nuksin": "KndPtSg1", #tingiv kõneviis minevik 1. isik ainsus aktiiv jaatav kõne
    "nuksite": "KndPtPl2", #tingiv kõneviis minevik 2. isik mitmus aktiiv jaatav kõne
    "nuvat": "KvtPtPs", #kaudne kõneviis minevik 1./2./3. isik ainsus/mitmus aktiiv jaatav kõne
    "o": "ImpPrSg2", #käskiv kõneviis olevik 2. isik ainsus aktiiv jaatav kõne
    "s": "IndIpfSg3", #kindel kõneviis lihtminevik 3. isik ainsus aktiiv jaatav kõne
    "sid": "IndIpfSg2|IndIpfPl3", #kindel kõneviis lihtminevik 2. isik ainsus aktiiv jaatav kõne / kindel kõneviis lihtminevik 3. isik mitmus aktiiv jaatav kõne
    "sime": "IndIpfPl1", #kindel kõneviis lihtminevik 1. isik mitmus aktiiv jaatav kõne
    "sin": "IndIpfSg1", #kindel kõneviis lihtminevik 1. isik ainsus aktiiv jaatav kõne
    "site": "IndIpfPl2", #kindel kõneviis lihtminevik 2. isik mitmus aktiiv jaatav kõne	
    "ta": "IndPrIps_", #kindel kõneviis olevik passiiv eitav kõne
    "tagu": "ImpPrIps", #käskiv kõneviis olevik passiiv jaatav kõne
    "taks": "KndPrIps", #tingiv kõneviis olevik passiiv jaatav kõne
    "takse": "IndPrIps", #kindel kõneviis olevik passiiv jaatav kõne
    "tama": "SupIps", #supiin passiiv jaatav kõne
    "tav": "PtsPrIps", #kesksõna olevik passiiv jaatav kõne
    "tavat": "KvtPrIps", #kaudne kõneviis olevik passiiv jaatav kõne
    "te": "IndPrPl2", #kindel kõneviis olevik 2. isik mitmus aktiiv jaatav kõne
    "ti": "IndIpfIps", #kindel kõneviis lihtminevik passiiv jaatav kõne
    "tud": "PtsPtIps", #kesksõna minevik passiiv jaatav kõne
    "tuks": "KndPtIps", #tingiv kõneviis minevik passiiv jaatav kõne
    "tuvat": "KvtPtIps", #kaudne kõneviis minevik passiiv jaatav kõne
    "v": "PtsPrPs", #kesksõna olevik aktiiv jaatav kõne
    "vad": "IndPrPl3", #kindel kõneviis olevik 3. isik mitmus aktiiv jaatav kõne
    "vat": "KvtPrPs", #kaudne kõneviis olevik 1./2./3. isik ainsus/mitmus aktiiv jaatav kõne
    "": "",
    "?": "?" # nt numeraalid
}