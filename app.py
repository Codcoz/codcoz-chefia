from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import (
     ChatPromptTemplate,
     MessagesPlaceholder,
     HumanMessagePromptTemplate,
     AIMessagePromptTemplate,
     FewShotChatMessagePromptTemplate
)
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os
from dotenv import load_dotenv
from pytz import timezone
from mongo_tools import RECEITAS_TOOLS
from pg_tools import TAREFAS_TOOLS
from flask import Flask, request, jsonify

load_dotenv()

app = Flask(__name__)

# Dicionário para armazenar o histórico de mensgens
store = {}

def get_session_history(session_id) -> ChatMessageHistory:
    # Função que retorna o histórico de uma sessão específica.
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

TZ = timezone("America/Sao_Paulo")
today = datetime.now(TZ).date()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0.7,
    top_p=0.95,
    api_key=GEMINI_API_KEY
)

llm_fast = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    temperature=0, # decisões determinísticas e agregação objetiva
    api_key=GEMINI_API_KEY
)

# prompt do agente roteador
system_prompt_roteador = ("system",
    """
    ### PERSONA SISTEMA
    - Você é o ChefIA, o assistente virtual dos gestores da cozinha. É objetivo, responsável, confiável e empático, com foco em utilidade imediata. Seu objetivo é ser um parceiro confiável para o usuário, auxiliando-o a tomar decisões relacionadas a gestão do estoque e gastronomia.
    - Evite jargões.
    - Evite ser prolixo.
    - Não invente dados.
    - Respostas não precisam ser necessariamente curtas, mas procure não falar demais.
    - Hoje é {today} (America/Sao_Paulo). Interprete datas relativas a partir desta data.


    ### PAPEL
    - Acolher o usuário e manter o foco em ESTOQUE ou RECEITAS da empresa.
    - Decidir a rota: {{receitas | tarefas}}.
    - Responder diretamente em:
    (a) saudações/small talk, ou 
    (b) fora de escopo (redirecionando para receitas/tarefas).
    - Seu objetivo é conversar de forma amigável com o usuário e tentar identificar se ele menciona algo sobre tarefas ou receitas.
    - Em fora_escopo: ofereça 1–2 sugestões práticas para voltar ao seu escopo (ex.: agendar/registrar tarefas, consultar/resumir receitas).
    - Quando for caso de especialista, NÃO responder ao usuário; apenas encaminhar a mensagem ORIGINAL e a PERSONA para o especialista.


    ### REGRAS
    - Seja breve, educado e objetivo.
    - Se faltar um dado absolutamente essencial para decidir a rota, faça UMA pergunta mínima (CLARIFY). Caso contrário, deixe CLARIFY vazio.
    - Responda de forma textual.


    ### PROTOCOLO DE ENCAMINHAMENTO (texto puro)
    ROUTE=<especialista>
    PERGUNTA_ORIGINAL=<mensagem completa do usuário, sem edições>
    PERSONA=<copie o bloco "PERSONA SISTEMA" daqui>
    CLARIFY=<pergunta mínima se precisar; senão deixe vazio>


    ### SAÍDAS POSSÍVEIS
    - Resposta direta (texto curto) quando for saudação.
    - Encaminhamento ao especialista usando exatamente o protocolo acima.


    ### HISTÓRICO DA CONVERSA
    {chat_history}


    ### CONTEXTO
    - ID da empresa: {empresa_id}
    - 
    """
)

example_prompt_base = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate.from_template("{human}"),
    AIMessagePromptTemplate.from_template("{ai}"),
])

shots_roteador = [
    # 1) Saudação -> resposta direta
    {
        "human": "Oi, tudo bem?",
        "ai": "Olá! Posso te ajudar com quais consultas hoje?"
    },
    # 2) Fora de escopo -> recusar e redirecionar
    {
        "human": "Me conta uma piada.",
        "ai": "Consigo ajudar apenas com gestão de estoque ou receitas. Prefere consultar receitas ou ver o estado do estoque?"
    },
    # 3) Tarefas -> encaminhar ao especialista (protocolo textual)
    {
        "human": "Adicione uma tarefa de Conferência de Estoque para Bruno Galvão.",
        "ai": "ROUTE=tarefas\nPERGUNTA_ORIGINAL=Adicione uma tarefa de Conferência de Estoque para Bruno Galvão.\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
    },
    # 4) Receitas -> encaminhar ao especialista (protocolo textual)
    {
        "human": "Quero fazer um hambúrguer. Quais receitas você me recomenda?",
        "ai": "ROUTE=receitas\nPERGUNTA_ORIGINAL=Quero fazer um hambúrguer. Quais receitas você me recomenda?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY="
    },
    # 5) Quem é você? -> resposta direta
    {
        "human": "Quem é você?",
        "ai": "Eu sou o ChefIA, seu assistente virtual para ajudar com receitas e estoques da sua empresa. Como posso ajudar hoje?"
    }
]

fewshots_roteador = FewShotChatMessagePromptTemplate(
    examples=shots_roteador,
    example_prompt=example_prompt_base
)

# -------------------
# - PROMPTS ESPECIALISTAS --------------------
system_prompt_receitas = ("system",
    """
    ### OBJETIVO
    Interpretar a PERGUNTA_ORIGINAL sobre consultas e operar as tools de 'receitas' para responder.
    A saída SEMPRE é JSON (contrato abaixo).


    ### TAREFAS
    - Consultar receitas de acordo com a pergunta do usuário.
    - SOMENTE CONSULTAS, você não vai inserir nada no MongoDB.
    - Resumir receitas disponíveis, baseados no nome.


    ### CONTEXTO
    - Hoje é {today} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
    - Entrada vem do Roteador via protocolo:
    - ROUTE=especialista
    - PERGUNTA_ORIGINAL=...
    - PERSONA=...   (use como diretriz de concisão/objetividade)
    - CLARIFY=...   (se preenchido, priorize responder esta dúvida antes de prosseguir)
    - ID da empresa: {empresa_id}
    - Todas as queries ao MongoDB devem ter como filtro obrigatório o ID da empresa.
    - Ao invocar ferramentas de consulta, SEMPRE utilize os valores de 'empresa_id' fornecidos no contexto para filtrar os dados.

    
    ### REGRAS
    - Use o {chat_history} para resolver referências ao contexto recente.


    ### SAÍDA (JSON)
    Campos mínimos para enviar para o roteador de volta:
    # Obrigatórios:
    - dominio      : "especialista"
    - intencao     : "consultar" | "resumo"
    - resposta     : uma frase objetiva e apontamentos
    - recomendacao : procurar por outras receitas ou buscar o passo a passo (pode ser string vazia se não houver)
    # Opcionais (incluir só se necessário):
    - acompanhamento : texto curto de follow-up/próximo passo
    - esclarecer     : pergunta mínima de clarificação (usar OU 'acompanhamento')
    - indicadores    : {{chaves livres e numéricas úteis ao log}}


    ### HISTÓRICO DA CONVERSA
    {chat_history}                                
    """
)

# Especialista receitas (mesmo example_prompt_pair)
shots_receitas = [
    {
        "human": "ROUTE=receitas\nPERGUNTA_ORIGINAL=Quero fazer um hambúrguer. Quais receitas você me recomenda?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"receitas","intencao":"consultar","resposta":"Você pode fazer um Hambúrguer de costela Maturatta com vinagrete, ótima opção para os amantes de hambúrguer. Segue o passo a passo:
        - Misture o alho, salsinha, pimenta, orégano, azeite, suco de limão, raspas, sal e pimenta para fazer o chimichurri;
        - Toste o pimentão em fogo alto até queimar a pele, colocar em um saco de plástico fechado por alguns minutos e limpar a pele queimada, porcionar em um quadrado um pouco maior que o pão;
        - Toste as fatias de bacon até ficarem crocantes;
        - Passe manteiga no pão e tostar ele na grelha;
        - Coloque sal e pimenta do reino no burger e levar ele à grelha com fogo muito alto, virar e servir ao ponto para menos.
        - Para a montagem, sobre o pão, coloque o pimentão, o burguer, bacon, chimichurri e o topo do pão.","recomendacao":"Quer conhecer outras receitas de hambúrguer?"}}"""
    },
    {
        "human": "ROUTE=receitas\nPERGUNTA_ORIGINAL=Que massa eu posso fazer hoje?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"receitas","intencao":"consultar","resposta":"Uma boa pedida seria o Ossobuco com nhoque ao limone.","acompanhamento":"Quer saber os ingredientes e o passo a passo?", "recomendacao":""}}"""
    },
    {
        "human": "ROUTE=receitas\nPERGUNTA_ORIGINAL=Quero um resumo dos tipos de receitas disponíveis\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"receitas","intencao":"resumo","resposta":"Você tem acesso a receitas para churrasco, café da manhã, almoço rápido, jantar caprichado, entre outros. Usando desde ingredientes mais comuns que todos tem em casa até os mais sofisticados.","recomendacao":"Quer conhecer alguma receita em específico?"}}"""
    },
]

fewshots_receitas = FewShotChatMessagePromptTemplate(
    examples=shots_receitas,
    example_prompt=example_prompt_base,
)

############################
# prompt do agente de agenda
system_prompt_tarefas = ("system",
    """
    ### OBJETIVO
    Interpretar a PERGUNTA_ORIGINAL sobre tarefas e (quando houver tools) consultar/criar/atualizar/excluir tarefas. 
    A saída SEMPRE é JSON (contrato abaixo) para o Orquestrador.


    ### TAREFAS
    - Processar perguntas do usuário sobre tarefas.
    - Analise entradas de tarefas informadas pelo usuário.
    - Oferecer dicas personalizadas sobre gestão de tarefas de funcionários.
    - Consultar histórico de tarefas quando relevante.


    ### CONTEXTO
    - Hoje é {today} (America/Sao_Paulo). Interprete datas relativas a partir desta data.
    - Entrada do Roteador:
    - ROUTE=tarefas
    - PERGUNTA_ORIGINAL=...
    - PERSONA=...   (use como diretriz de concisão/objetividade)
    - CLARIFY=...   (se preenchido, responda primeiro)
    - ID da empresa: {empresa_id}
    - ID do gestor: {gestor_id}
    - Todas as queries feita no PostgreSQL devem ter como filtro obrigatório o ID da empresa.
    - Ao invocar ferramentas de consulta, SEMPRE utilize os valores de 'empresa_id' fornecidos no contexto para filtrar os dados.

    ### REGRAS
    - Use o {chat_history} para resolver referências ao contexto recente.


    ### SAÍDA (JSON)
    # Obrigatórios:
     - dominio   : "agenda"
     - intencao  : "consultar" | "criar" | "atualizar" | "cancelar" | "listar"
     - resposta  : uma frase objetiva
     - recomendacao : ação prática (pode ser string vazia)
    # Opcionais (incluir só se necessário):
     - acompanhamento : texto curto de follow-up/próximo passo
     - esclarecer     : pergunta mínima de clarificação
     - janela_tempo   : {{"de":"YYYY-MM-DD","ate":"YYYY-MM-DD","rotulo":"ex.: semana que vem"}}


     ### HISTÓRICO DA CONVERSA
    {chat_history}
    """
)

shots_tarefas = [
    # 1) Tarefa - criar
    {
        "human": "ROUTE=tarefas\nPERGUNTA_ORIGINAL=Adicione uma tarefa de Conferência de Estoque para Bruno Galvão.\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"tarefas","intencao":"criar","resposta":"Tarefa adicionada para Bruno Galvão.","recomendacao":""}}"""
    },
    # 2) Tarefa - consultar
    {
        "human": "ROUTE=tarefas\nPERGUNTA_ORIGINAL=Quais são as tarefas agendadas para semana que vem?\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"tarefas","intencao":"consultar","resposta":"São 2 tarefas agendadas para o Bruno Galvão e 1 para Raissa Casale na semana que vem.","recomendacao":"Quer cancelar uma delas ou agendar mais alguma?","janela_tempo":{{"de":"2025-10-17","ate":"2025-10-21","rotulo":"semana que vem"}} }}"""
    },
    # 3) Tarefa - cancelar
    {
        "human": "ROUTE=tarefas\nPERGUNTA_ORIGINAL=Cancele as tarefas da semana que vem para Gabriel Koji.\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"tarefas","intencao":"cancelar","resposta":"Tarefas do dia 1/11/2025 e 3/11/2025 canceladas.","recomendacao":""}}"""
    },
    # 4) Tarefa - falta dado -> esclarecer
    {
        "human": "ROUTE=tarefas\nPERGUNTA_ORIGINAL=Adicione uma tarefa.\nPERSONA={PERSONA_SISTEMA}\nCLARIFY=",
        "ai": """{{"dominio":"tarefas","intencao":"criar","resposta":"Preciso do responsável que irá realizar a tarefa para prosseguir.","recomendacao":"", "esclarecer": "Quem será o responsável dessa tarefa?"}}"""
    },
]

fewshots_tarefas = FewShotChatMessagePromptTemplate(
    examples=shots_tarefas,
    example_prompt=example_prompt_base,
)

### Agente orquestrador ####
system_prompt_orquestrador = ("system",
    """
### PAPEL
Você é o Agente Orquestrador do ChefIA. Sua função é entregar a resposta final ao usuário **somente** quando um Especialista retornar o JSON.


### ENTRADA
- ESPECIALISTA_JSON contendo chaves como: dominio, intencao, resposta, recomendacao, acompanhamento (opcional), esclarecer (opcional), indicadores (opcional)


### REGRAS
- Use **exatamente** `resposta` do especialista como a **primeira linha** do output.
- Se `recomendacao` existir e não for vazia, inclua a seção *Recomendação*; caso contrário, **omita**.
- Para *Acompanhamento*: se houver `esclarecer`, use-o; senão, se houver `acompanhamento`, use-o; caso contrário, **omita** a seção.
- Não reescreva números/datas se já vierem prontos. Não invente dados. Seja conciso.
- Não retorne JSON; **sempre** retorne no FORMATO DE SAÍDA.


### FORMATO DE SAÍDA (sempre ao usuário)
<sua resposta será 1 frase objetiva sobre a situação>
- *Recomendação*:
<ação prática e imediata>     # omita esta seção se não houver recomendação
- *Acompanhamento* (opcional):
<pergunta/mini próximo passo>  # omita se nada for necessário


### HISTÓRICO DA CONVERSA
{chat_history}
"""
)

shots_orquestrador = [
    # 1) Receitas — resumo
    {
        "human": """ESPECIALISTA_JSON:\n{{"dominio":"receitas","intencao":"resumo","resposta":"Você tem acesso a receitas para churrasco, almoço rápido, jantar caprichado, entre outros. Usando desde ingredientes mais comuns que todos tem em casa até os mais sofisticados.","recomendacao":"Quer conhecer alguma receita em específico?"}}""",
        "ai": "Você tem acesso a receitas para churrasco, café da manhã, almoço rápido, jantar caprichado, entre outros. Usando desde ingredientes mais comuns que todos tem em casa até os mais sofisticados.\n- *Recomendação*:\n Quer conhecer alguma receita em específico?"
    },

    # 2) Receitas — consultar
    {
        "human": """ESPECIALISTA_JSON:\n{{"dominio":"receitas","intencao":"consultar","resposta":"Uma boa pedida seria o Ossobuco com nhoque ao limone.","recomendacao":"","acompanhamento":"Quer saber os ingredientes e o passo a passo?"}}""",
        "ai": """Uma boa pedida seria o Ossobuco com nhoque ao limone.\n- *Acompanhamento* (opcional):\nQuer saber os ingredientes e o passo a passo?"""
    },

    # 3) Tarefa — falta dado -> esclarecer
    {
        "human": """ESPECIALISTA_JSON:\n{{"dominio":"tarefa","intencao":"criar","resposta":"Preciso do responsável que irá realizar a tarefa para prosseguir.","recomendacao":"", "esclarecer": "Quem será o responsável dessa tarefa?"}}""",
        "ai": """Preciso do responsável que irá realizar a tarefa para prosseguir. Quem será o responsável dessa tarefa?"""
    },
]

fewshots_orquestrador = FewShotChatMessagePromptTemplate(
    examples=shots_orquestrador,
    example_prompt=example_prompt_base,
)

prompt_roteador = ChatPromptTemplate.from_messages([
    system_prompt_roteador,                 # system prompt
    fewshots_roteador,                      # Shots human/ai 
    MessagesPlaceholder("chat_history"),    # memória
    ("human", "{input}"),                   # user prompt
]).partial(today=today.isoformat(), empresa_id="{empresa_id}", gestor_id="{gestor_id}")   # Com partial(), você injeta valores fixos que ficam pré-preenchidos no template.

prompt_orquestrador = ChatPromptTemplate.from_messages([
    system_prompt_orquestrador,             # system prompt
    fewshots_orquestrador,                  # Shots human/ai 
    MessagesPlaceholder("chat_history"),    # memória
    ("human", "{input}"),                   # user prompt
]).partial(today=today.isoformat())   # Com partial(), você injeta valores fixos que ficam pré-preenchidos no template.

prompt_receitas = ChatPromptTemplate.from_messages([
    system_prompt_receitas,                 # system prompt
    fewshots_receitas,                      # Shots human/ai 
    MessagesPlaceholder("chat_history"),    # memória
    ("human", "{input}"),                   # user prompt
    MessagesPlaceholder("agent_scratchpad") # espaço reservado para pensamentos internos do LLM (chain of thought)
]).partial(today=today.isoformat(), empresa_id="{empresa_id}")   # Com partial(), você injeta valores fixos que ficam pré-preenchidos no template.

prompt_tarefas = ChatPromptTemplate.from_messages([
    system_prompt_tarefas,                  # system prompt
    fewshots_tarefas,                       # Shots human/ai 
    MessagesPlaceholder("chat_history"),    # memória
    ("human", "{input}"),                   # user prompt
    MessagesPlaceholder("agent_scratchpad") # espaço reservado para pensamentos internos do LLM (chain of thought)
]).partial(today=today.isoformat(), empresa_id="{empresa_id}", gestor_id="{empresa_id}")   # Com partial(), você injeta valores fixos que ficam pré-preenchidos no template.

# Instanciamento de agentes COM acesso A TOOLS
receitas_agent = create_tool_calling_agent(llm, RECEITAS_TOOLS, prompt_receitas)
receitas_executor_base = AgentExecutor(
    agent=receitas_agent,
    tools=RECEITAS_TOOLS,
    verbose=False,
    handle_parsing_errors=True,
    return_intermediate_steps=False
)
receitas_executor = RunnableWithMessageHistory(
    receitas_executor_base,
    get_session_history=get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)

tarefas_agent = create_tool_calling_agent(llm, TAREFAS_TOOLS, prompt_tarefas)
tarefas_executor_base = AgentExecutor(agent=tarefas_agent, tools=TAREFAS_TOOLS, verbose=False, handle_parsing_errors=True, return_intermediate_steps=False)
tarefas_executor = RunnableWithMessageHistory(
    tarefas_executor_base,
    get_session_history=get_session_history,
    input_messages_key='input',
    history_messages_key='chat_history'
)

# Instanciamento de agentes SEM acesso A TOOLS
roteador_chain = RunnableWithMessageHistory(
    prompt_roteador | llm_fast | StrOutputParser(),
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

orquestrador_chain = RunnableWithMessageHistory(
    prompt_orquestrador | llm_fast | StrOutputParser(),
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

def executar_fluxo_chefia(pergunta_usuario, session_id, empresa_id, gestor_id):
    resp_roteador = roteador_chain.invoke(
            {
                "input": pergunta_usuario,
                "empresa_id": empresa_id,
                "gestor_id": gestor_id
            },
            config={'configurable': {'session_id': session_id}} # Aqui, entraria o ID do usuário e histórico.
        )
    
    if "ROUTE=" not in resp_roteador:
        return resp_roteador
    elif "ROUTE=receitas" in resp_roteador:
        resp_receitas = receitas_executor.invoke(
            {
                "input": resp_roteador,
                "empresa_id": empresa_id
            },
            config={'configurable': {'session_id': session_id}} # Aqui, entraria o ID do usuário e histórico.
        )

        resp_orquestrador = orquestrador_chain.invoke(
            {"input": resp_receitas["output"]},
            config={'configurable': {'session_id': session_id}} # Aqui, entraria o ID do usuário e histórico.
        )

        return resp_orquestrador
    elif "ROUTE=tarefas" in resp_roteador:
        resp_tarefas = receitas_executor.invoke(
            {
                "input": resp_roteador,
                "empresa_id": empresa_id,
                "gestor_id": gestor_id
            },
            config={'configurable': {'session_id': session_id}} # Aqui, entraria o ID do usuário e histórico.
        )

        resp_orquestrador = orquestrador_chain.invoke(
            {"input": resp_tarefas["output"]},
            config={'configurable': {'session_id': session_id}} # Aqui, entraria o ID do usuário e histórico.
        )

        return resp_orquestrador

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Dados não fornecidos ou formato inválido!"}), 400
    
    user_message = data.get("user_message", "")
    empresa_id = data.get("empresa_id", "")
    gestor_id = data.get("gestor_id", "")

    if not user_message:
        return jsonify({"error": "A mensagem do usuário está vazia!"}), 400
    
    try:
        resposta = executar_fluxo_chefia(
            pergunta_usuario=user_message,
            session_id="",
            empresa_id=empresa_id,
            gestor_id=gestor_id
        )
        return jsonify({"status": "ok", "resposta": resposta}), 200

    except Exception as e:
        print(f"Erro no fluxo: {e}")
        return jsonify({"status": "error", "resposta": "Erro ao processar a solicitação."}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    app.run(debug=True)

# while True:
#     user_input = input('> ')
#     if user_input.lower() in ('sair', 'end', 'fim', 'tchau', 'bye'):
#         print('Encerrando a conversa...')
#         break
#     try:
        
#         # Chama a função orquestradora que executa o fluxo completo (Roteador -> Especialista -> Orquestrador)
#         resposta = executar_fluxo_chefia(
#             pergunta_usuario=user_input,
#             session_id="",
#             empresa_id=empresa_id,
#             gestor_id=gestor_id
#         )

#         print(resposta)
        
#     except Exception as e:
#         print('Erro ao consumir a API: ', e)