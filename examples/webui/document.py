import streamlit as st
import os
import time
import traceback
from typing import Literal, Dict, Tuple
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd

from .utils import *
from muagent.utils.path_utils import *
from muagent.service.service_factory import get_kb_details, get_kb_doc_details
from muagent.orm import table_init

from configs.model_config import (
    KB_ROOT_PATH, kbs_config, DEFAULT_VS_TYPE, WEB_CRAWL_PATH,
    EMBEDDING_DEVICE, EMBEDDING_ENGINE, EMBEDDING_MODEL, embedding_model_dict,
    llm_model_dict, model_engine, em_apikey, em_apiurl
)

# SENTENCE_SIZE = 100

cell_renderer = JsCode("""function(params) {if(params.value==true){return '鉁�'}else{return '脳'}}""")


def config_aggrid(
        df: pd.DataFrame,
        columns: Dict[Tuple[str, str], Dict] = {},
        selection_mode: Literal["single", "multiple", "disabled"] = "single",
        use_checkbox: bool = False,
) -> GridOptionsBuilder:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("No", width=40)
    for (col, header), kw in columns.items():
        gb.configure_column(col, header, wrapHeaderText=True, **kw)
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=use_checkbox,
        # pre_selected_rows=st.session_state.get("selected_rows", [0]),
    )
    return gb


def file_exists(kb: str, selected_rows: List) -> Tuple[str, str]:
    '''
    check whether a doc file exists in local knowledge base folder.
    return the file's name and path if it exists.
    '''
    values = selected_rows.to_dict('records') if isinstance(selected_rows, pd.DataFrame) else selected_rows
    if values:
        file_name = values[0]["file_name"]
        file_path = get_file_path(kb, file_name, KB_ROOT_PATH)
        if os.path.isfile(file_path):
            return file_name, file_path
    return "", ""


def knowledge_page(
        api: ApiRequest, 
        embedding_model_dict: dict = embedding_model_dict, 
        kbs_config: dict = kbs_config, 
        embedding_model: str = EMBEDDING_MODEL, 
        default_vs_type: str = DEFAULT_VS_TYPE, 
        web_crawl_path: str = WEB_CRAWL_PATH
        ):
    # 鍒ゆ柇琛ㄦ槸鍚﹀瓨鍦ㄥ苟杩涜鍒濆鍖�
    table_init()

    try:
        kb_list = {x["kb_name"]: x for x in get_kb_details(KB_ROOT_PATH)}
    except Exception as e:
        st.error("鑾峰彇鐭ヨ瘑搴撲俊鎭敊璇紝璇锋鏌ユ槸鍚﹀凡鎸夌収 `README.md` 涓� `4 鐭ヨ瘑搴撳垵濮嬪寲涓庤縼绉籤 姝ラ瀹屾垚鍒濆鍖栨垨杩佺Щ锛屾垨鏄惁涓烘暟鎹簱杩炴帴閿欒銆�")
        st.stop()
    kb_names = list(kb_list.keys())

    if "selected_kb_name" in st.session_state and st.session_state["selected_kb_name"] in kb_names:
        selected_kb_index = kb_names.index(st.session_state["selected_kb_name"])
    else:
        selected_kb_index = 0

    def format_selected_kb(kb_name: str) -> str:
        if kb := kb_list.get(kb_name):
            return f"{kb_name} ({kb['vs_type']} @ {kb['embed_model']})"
        else:
            return kb_name

    selected_kb = st.selectbox(
        "璇烽€夋嫨鎴栨柊寤虹煡璇嗗簱锛�",
        kb_names + ["鏂板缓鐭ヨ瘑搴�"],
        format_func=format_selected_kb,
        index=selected_kb_index
    )

    llm_config = LLMConfig(
        model_name=LLM_MODEL, 
        model_engine=model_engine,
        api_key=llm_model_dict[LLM_MODEL]["api_key"],
        api_base_url=llm_model_dict[LLM_MODEL]["api_base_url"],
    )
    embed_config = EmbedConfig(
        embed_model=EMBEDDING_MODEL, embed_model_path=embedding_model_dict[EMBEDDING_MODEL],
        model_device=EMBEDDING_DEVICE, embed_engine=EMBEDDING_ENGINE,
        api_key=em_apikey, api_base_url=em_apiurl
    )
    
    if selected_kb == "鏂板缓鐭ヨ瘑搴�":
        with st.form("鏂板缓鐭ヨ瘑搴�"):

            kb_name = st.text_input(
                "鏂板缓鐭ヨ瘑搴撳悕绉�",
                placeholder="鏂扮煡璇嗗簱鍚嶇О锛屼笉鏀寔涓枃鍛藉悕",
                key="kb_name",
            )

            cols = st.columns(2)

            vs_types = list(kbs_config.keys())
            vs_type = cols[0].selectbox(
                "鍚戦噺搴撶被鍨�",
                vs_types,
                index=vs_types.index(default_vs_type),
                key="vs_type",
            )

            embed_models = list(embedding_model_dict.keys())

            embed_model = cols[1].selectbox(
                "Embedding 妯″瀷",
                embed_models,
                index=embed_models.index(embedding_model),
                key="embed_model",
            )

            submit_create_kb = st.form_submit_button(
                "鏂板缓",
                # disabled=not bool(kb_name),
                use_container_width=True,
            )

        if submit_create_kb:
            if not kb_name or not kb_name.strip():
                st.error(f"鐭ヨ瘑搴撳悕绉颁笉鑳戒负绌猴紒")
            elif kb_name in kb_list:
                st.error(f"鍚嶄负 {kb_name} 鐨勭煡璇嗗簱宸茬粡瀛樺湪锛�")
            else:
                ret = api.create_knowledge_base(
                    knowledge_base_name=kb_name,
                    vector_store_type=vs_type,
                    llm_config=llm_config, embed_config=embed_config
                    # embed_model=embed_model,
                    # embed_engine=EMBEDDING_ENGINE,
                    # embedding_device= EMBEDDING_DEVICE,
                    # embed_model_path=embedding_model_dict[embed_model],
                    # api_key=llm_model_dict[LLM_MODEL]["api_key"],
                    # api_base_url=llm_model_dict[LLM_MODEL]["api_base_url"],
                )
                st.toast(ret.get("msg", " "))
                st.session_state["selected_kb_name"] = kb_name
                st.experimental_rerun()
                # st.rerun()

    elif selected_kb:
        kb = selected_kb

        # 涓婁紶鏂囦欢
        # sentence_size = st.slider("鏂囨湰鍏ュ簱鍒嗗彞闀垮害闄愬埗", 1, 1000, SENTENCE_SIZE, disabled=True)
        files = st.file_uploader("涓婁紶鐭ヨ瘑鏂囦欢",
                                 [i for ls in LOADER2EXT_DICT.values() for i in ls],
                                 accept_multiple_files=True,
                                 )

        if st.button(
                "娣诲姞鏂囦欢鍒扮煡璇嗗簱",
                # help="璇峰厛涓婁紶鏂囦欢锛屽啀鐐瑰嚮娣诲姞",
                # use_container_width=True,
                disabled=len(files) == 0,
        ):
            data = [
                {
                    "file": f, "knowledge_base_name": kb, "not_refresh_vs_cache": True, 
                    "llm_config": llm_config, "embed_config": embed_config
                }
                for f in files
            ]
            data[-1]["not_refresh_vs_cache"]=False
            for k in data:
                pass
                ret = api.upload_kb_doc(**k)
                if msg := check_success_msg(ret):
                    st.toast(msg, icon="鉁�")
                elif msg := check_error_msg(ret):
                    st.toast(msg, icon="鉁�")
            st.session_state.files = []

        base_url = st.text_input(
            "寰呰幏鍙栧唴瀹圭殑URL鍦板潃",
            placeholder="璇峰～鍐欐纭彲鎵撳紑鐨刄RL鍦板潃",
            key="base_url",
        )


        if st.button(
            "娣诲姞URL鍐呭鍒扮煡璇嗗簱",
            disabled= base_url is None or base_url=="",
            ):
            filename = base_url.replace("https://", " ").\
                replace("http://", " ").replace("/", " ").\
                replace("?", " ").replace("=", " ").replace(".", " ").strip()
            html_name = "_".join(filename.split(" ",) + ["html.jsonl"])
            text_name = "_".join(filename.split(" ",) + ["text.jsonl"])
            html_path = os.path.join(web_crawl_path, html_name,)
            text_path = os.path.join(web_crawl_path, text_name,)
            # if not os.path.exists(text_dir) or :
            st.toast(base_url)
            st.toast(html_path)
            st.toast(text_path)
            res = api.web_crawl(
                base_url=base_url,
                html_dir=html_path,
                text_dir=text_path,
                do_dfs = False,
                reptile_lib="requests",
                method="get",
                time_sleep=2,
                )
            
            if res["status"] == 200:
                st.toast(res["response"], icon="鉁�")
                data = [
                    {
                        "file": text_path, "filename": text_name, "knowledge_base_name": kb, 
                        "not_refresh_vs_cache": False, "llm_config": llm_config, "embed_config": embed_config
                    }
                ]
                for k in data:
                    ret = api.upload_kb_doc(**k)
                    logger.info(ret)
                    if msg := check_success_msg(ret):
                        st.toast(msg, icon="鉁�")
                    elif msg := check_error_msg(ret):
                        st.toast(msg, icon="鉁�")
                st.session_state.files = []
            else:
                st.toast(res["response"], icon="鉁�")

            if os.path.exists(html_path):
                os.remove(html_path)

        st.divider()

        # 鐭ヨ瘑搴撹鎯�
        # st.info("璇烽€夋嫨鏂囦欢锛岀偣鍑绘寜閽繘琛屾搷浣溿€�")
        doc_details = pd.DataFrame(get_kb_doc_details(kb, KB_ROOT_PATH))
        if not len(doc_details):
            st.info(f"鐭ヨ瘑搴� `{kb}` 涓殏鏃犳枃浠�")
        else:
            st.write(f"鐭ヨ瘑搴� `{kb}` 涓凡鏈夋枃浠�:")
            st.info("鐭ヨ瘑搴撲腑鍖呭惈婧愭枃浠朵笌鍚戦噺搴擄紝璇蜂粠涓嬭〃涓€夋嫨鏂囦欢鍚庢搷浣�")
            doc_details.drop(columns=["kb_name"], inplace=True)
            doc_details = doc_details[[
                "No", "file_name", "document_loader", "text_splitter", "in_folder", "in_db",
            ]]
            # doc_details["in_folder"] = doc_details["in_folder"].replace(True, "鉁�").replace(False, "脳")
            # doc_details["in_db"] = doc_details["in_db"].replace(True, "鉁�").replace(False, "脳")
            gb = config_aggrid(
                doc_details,
                {
                    ("No", "搴忓彿"): {},
                    ("file_name", "鏂囨。鍚嶇О"): {},
                    # ("file_ext", "鏂囨。绫诲瀷"): {},
                    # ("file_version", "鏂囨。鐗堟湰"): {},
                    ("document_loader", "鏂囨。鍔犺浇鍣�"): {},
                    ("text_splitter", "鍒嗚瘝鍣�"): {},
                    # ("create_time", "鍒涘缓鏃堕棿"): {},
                    ("in_folder", "婧愭枃浠�"): {"cellRenderer": cell_renderer},
                    ("in_db", "鍚戦噺搴�"): {"cellRenderer": cell_renderer},
                },
                "multiple",
            )

            doc_grid = AgGrid(
                doc_details,
                gb.build(),
                columns_auto_size_mode="FIT_CONTENTS",
                theme="alpine",
                custom_css={
                    "#gridToolBar": {"display": "none"},
                },
                allow_unsafe_jscode=True
            )

            selected_rows = doc_grid.get("selected_rows", [])

            cols = st.columns(4)
            file_name, file_path = file_exists(kb, selected_rows)
            if file_path:
                with open(file_path, "rb") as fp:
                    cols[0].download_button(
                        "涓嬭浇閫変腑鏂囨。",
                        fp,
                        file_name=file_name,
                        use_container_width=True, )
            else:
                cols[0].download_button(
                    "涓嬭浇閫変腑鏂囨。",
                    "",
                    disabled=True,
                    use_container_width=True, )

            st.write()
            # 灏嗘枃浠跺垎璇嶅苟鍔犺浇鍒板悜閲忓簱涓�
            row_values = selected_rows.to_dict('records') if isinstance(selected_rows, pd.DataFrame) else selected_rows
            if cols[1].button(
                    "閲嶆柊娣诲姞鑷冲悜閲忓簱" if row_values and (pd.DataFrame(row_values)["in_db"]).any() else "娣诲姞鑷冲悜閲忓簱",
                    disabled=not file_exists(kb, row_values)[0],
                    use_container_width=True,
            ):
                for row in row_values:
                    api.update_kb_doc(kb, row["file_name"], llm_config=llm_config, embed_config=embed_config,
                                    #   embed_engine=EMBEDDING_ENGINE,embed_model=EMBEDDING_MODEL,
                                    #   embed_model_path=embedding_model_dict[EMBEDDING_MODEL],
                                    #   model_device=EMBEDDING_DEVICE,
                                    #   api_key=llm_model_dict[LLM_MODEL]["api_key"],
                                    #   api_base_url=llm_model_dict[LLM_MODEL]["api_base_url"],
                                      )
                st.experimental_rerun()
                #st.rerun()

            # 灏嗘枃浠朵粠鍚戦噺搴撲腑鍒犻櫎锛屼絾涓嶅垹闄ゆ枃浠舵湰韬€�
            if cols[2].button(
                    "浠庡悜閲忓簱鍒犻櫎",
                    disabled=not (row_values and row_values[0]["in_db"]),
                    use_container_width=True,
            ):
                for row in row_values:
                    api.delete_kb_doc(kb, row["file_name"],
                                      llm_config=llm_config, embed_config=embed_config,)
                                    #   embed_engine=EMBEDDING_ENGINE,embed_model=EMBEDDING_MODEL,
                                    #   embed_model_path=embedding_model_dict[EMBEDDING_MODEL],
                                    #   model_device=EMBEDDING_DEVICE,
                                    #   api_key=llm_model_dict[LLM_MODEL]["api_key"],
                                    #   api_base_url=llm_model_dict[LLM_MODEL]["api_base_url"],)
                st.experimental_rerun()
                # st.rerun()

            if cols[3].button(
                    "浠庣煡璇嗗簱涓垹闄�",
                    type="primary",
                    use_container_width=True,
            ):
                for row in row_values:
                    ret = api.delete_kb_doc(kb, row["file_name"], True,
                                            llm_config=llm_config, embed_config=embed_config,)
                                    #   embed_engine=EMBEDDING_ENGINE,embed_model=EMBEDDING_MODEL,
                                    #   embed_model_path=embedding_model_dict[EMBEDDING_MODEL],
                                    #   model_device=EMBEDDING_DEVICE,
                                    #   api_key=llm_model_dict[LLM_MODEL]["api_key"],
                                    #   api_base_url=llm_model_dict[LLM_MODEL]["api_base_url"],)
                    st.toast(ret.get("msg", " "))
                st.experimental_rerun()
                #st.rerun()

        st.divider()

        cols = st.columns(3)

        # todo: freezed
        if cols[0].button(
                "渚濇嵁婧愭枃浠堕噸寤哄悜閲忓簱",
                # help="鏃犻渶涓婁紶鏂囦欢锛岄€氳繃鍏跺畠鏂瑰紡灏嗘枃妗ｆ嫹璐濆埌瀵瑰簲鐭ヨ瘑搴揷ontent鐩綍涓嬶紝鐐瑰嚮鏈寜閽嵆鍙噸寤虹煡璇嗗簱銆�",
                use_container_width=True,
                type="primary",
        ):
            with st.spinner("鍚戦噺搴撻噸鏋勪腑锛岃鑰愬績绛夊緟锛屽嬁鍒锋柊鎴栧叧闂〉闈€€�"):
                empty = st.empty()
                empty.progress(0.0, "")
                for d in api.recreate_vector_store(
                    kb, vs_type=default_vs_type, embed_model=embedding_model, embedding_device=EMBEDDING_DEVICE,
                      embed_model_path=embedding_model_dict[embedding_model], embed_engine=EMBEDDING_ENGINE,
                      api_key=llm_model_dict[LLM_MODEL]["api_key"],
                      api_base_url=llm_model_dict[LLM_MODEL]["api_base_url"],
                    ):
                    if msg := check_error_msg(d):
                        st.toast(msg)
                    else:
                        empty.progress(d["finished"] / d["total"], f"姝ｅ湪澶勭悊锛� {d['doc']}")
                st.experimental_rerun()
                # st.rerun()

        if cols[2].button(
                "鍒犻櫎鐭ヨ瘑搴�",
                use_container_width=True,
        ):
            ret = api.delete_knowledge_base(kb,)
            st.toast(ret.get("msg", " "))
            time.sleep(1)
            st.experimental_rerun()
            # st.rerun()
