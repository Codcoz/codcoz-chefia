import os
from dotenv import load_dotenv
import psycopg2
from typing import Optional, List
from langchain.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

def get_conn():
    return psycopg2.connect(os.getenv("SQL_URL"))

# Essa classe garante que o objeto no Python passe todos esses campos
class AddTarefaArgs(BaseModel):
    responsavel: str = Field(..., description="Nome ou nome completo da pessoa que realizou/irá realizar a tarefa.")
    empresa_id: int = Field(..., description="ID da empresa para filtrar as tarefas.")
    situacao: str = Field(..., description="Situação da tarefa: 'PENDENTE' | 'CONCLUÍDA' | 'CANCELADA'. Se ausente, cadastra como 'PENDENTE'.")
    ingrediente: Optional[str] = Field(default=None, description="Ingrediente que será consultado na tarefa.")
    pedido_id: Optional[int] = Field(default=None, description="ID do pedido associado a tarefa.")
    data_conclusao: Optional[str] = Field(default=None, description="Data que a tarefa foi concluída, passe no formato 'YYYY-MM-DD'.")
    gestor_id: Optional[str] = Field(default=None, description="ID da pessoa que está atribuindo a tarefa.")
    tipo_tarefa: Optional[str] = Field(default=None, description="Tipo da tarefa que será realizada.")
    data_limite: Optional[str] = Field(default=None, description="Data limite para conclusão da tarefa, passe no formato 'YYYY-MM-DD'.")

class CancelTarefaArgs(BaseModel):
    empresa_id: int = Field(..., description="ID da empresa para filtrar as tarefas.")
    situacao: str = Field(..., description="Situação da tarefa: 'PENDENTE' | 'CONCLUÍDA' | 'CANCELADA'. Se ausente, atualiza como 'CANCELADA'.")
    responsavel: Optional[str] = Field(default=None, description="Nome ou nome completo da pessoa que realizou/irá realizar a tarefa.")
    ingrediente: Optional[str] = Field(default=None, description="Ingrediente que será consultado na tarefa.")
    pedido_id: Optional[int] = Field(default=None, description="ID do pedido associado a tarefa.")
    tipo_tarefa: Optional[str] = Field(default=None, description="Tipo da tarefa que será realizada.")
    data_inicio_limite: Optional[str] = Field(default=None, description="Data limite início do filtro de tarefas, passe no formato 'YYYY-MM-DD'.") 
    data_fim_limite: Optional[str] = Field(default=None, description="Data limite final do filtro de tarefas, passe no formato 'YYYY-MM-DD'.") 
    data_limite: Optional[str] = Field(default=None, description="Data limite para conclusão da tarefa, passe no formato 'YYYY-MM-DD'.")
    data_inicio_criacao: Optional[str] = Field(default=None, description="Data de criação final do filtro de tarefas, passe no formato 'YYYY-MM-DD'.") 
    data_fim_criacao: Optional[str] = Field(default=None, description="Data de criação final do filtro de tarefas, passe no formato 'YYYY-MM-DD'.") 
    data_criacao: Optional[str] = Field(default=None, description="Data de criação da tarefa, passe no formato 'YYYY-MM-DD'.")

#Funções para garantir que os campos normalizados recebam IDs válidos
def get_ingrediente_id(cursor, nome_ingrediente: Optional[str]) -> Optional[int]:
    if not nome_ingrediente:
        return None
    cursor.execute(
        "SELECT id FROM ingrediente WHERE nome ILIKE %s LIMIT 1;", 
        (f"%{nome_ingrediente.replace(" ", "%")}%", )
    )
    row = cursor.fetchone()
    return row[0] if row else None

def get_tipo_tarefa_id(cursor, tipo_tarefa: Optional[str]) -> Optional[int]:
    if not tipo_tarefa:
        return None
    cursor.execute(
        "SELECT id FROM tipo_tarefa WHERE nome ILIKE %s OR descricao ILIKE %s LIMIT 1;", 
        (f"%{tipo_tarefa.replace(" ", "%")}%", f"%{tipo_tarefa.replace(" ", "%")}%", )
    )
    row = cursor.fetchone()
    return row[0] if row else 2 # tipo de tarefa 'padrão' 

def get_responsavel_id(cursor, responsavel: str) -> int:
    cursor.execute(
        "SELECT id FROM funcionario WHERE nome ILIKE %s OR concat(nome, ' ', sobrenome) ILIKE %s LIMIT 1;",
        (f"%{responsavel.replace(" ", "%")}%",f"%{responsavel.replace(" ", "%")}%" , )  
    )
    row = cursor.fetchone()
    return row[0] if row else None

@tool("add_tarefa", args_schema=AddTarefaArgs)
def add_tarefa(
    responsavel: str,
    empresa_id: int,
    situacao: str,
    ingrediente: Optional[str] = None,
    pedido_id: Optional[int] = None,
    data_conclusao: Optional[str] = None,
    gestor_id: Optional[int] = None,
    tipo_tarefa: Optional[str] = None,
    data_limite : Optional[str] = None,
) -> dict:
    """Adiciona/cria uma tarefa no banco de dados PostgreSQL.""" # docstring obrigatório da @tools do langchain (estranho, mas legal né?)
    conn = get_conn()
    cur = conn.cursor()

    try:
        ingrediente_id = get_ingrediente_id(cur, ingrediente)
        tipo_tarefa_id = get_tipo_tarefa_id(cur, tipo_tarefa)
        responsavel_id = get_responsavel_id(cur, responsavel)

        insert_tarefa = """
        INSERT INTO tarefa ( empresa_id, tipo_tarefa_id, ingrediente_id, relator_id, responsavel_id, pedido_id, situacao, data_limite, data_conclusao, data_criacao)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE) RETURNING id, data_criacao;        
        """

        cur.execute(insert_tarefa, (empresa_id, tipo_tarefa_id, ingrediente_id, gestor_id, responsavel_id, pedido_id, situacao, data_limite, data_conclusao))

        new_id, occurred = cur.fetchone()
        conn.commit()
        return {"status": "ok", "id": new_id, "occurred_at": str(occurred)}

    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass

@tool("cancel_tarefas", args_schema=CancelTarefaArgs)
def cancel_tarefas(
    empresa_id: int,
    situacao: str,
    responsavel: Optional[str] = None,
    ingrediente: Optional[str] = None,
    pedido_id: Optional[int] = None,
    tipo_tarefa: Optional[str] = None,
    data_inicio_limite: Optional[str] = None,
    data_fim_limite: Optional[str] = None,
    data_limite: Optional[str] = None,
    data_inicio_criacao: Optional[str] = None,
    data_fim_criacao: Optional[str] = None,
    data_criacao: Optional[str] = None,
) -> dict:
    """
    Cancela as tarefas com filtros por responsável, gestor, tipo da tarefa, data de criação, data de limite e datas locais (America/Sao_Paulo).
    """
    try:
        conn = get_conn()
        cur = conn.cursor()
                
        sql_statement = f"""
            UPDATE tarefa
               SET situacao = '{situacao}'
             WHERE empresa_id = {empresa_id} 
        """

        if tipo_tarefa:
            tipo_tarefa_id = get_tipo_tarefa_id(cur, tipo_tarefa)
            sql_statement += f" AND tipo_tarefa_id = {tipo_tarefa_id}"

        if responsavel:
            responsavel_id = get_responsavel_id(cur, responsavel)
            sql_statement += f" AND responsavel_id = {responsavel_id}"

        if ingrediente:
            ingrediente_id = get_ingrediente_id(cur, ingrediente)
            sql_statement += f" AND ingrediente_id = {ingrediente_id}"
        
        if pedido_id:
            sql_statement += f" AND pedido_id = {pedido_id}"
        
        if data_limite:
            sql_statement += f" AND data_limite = '{data_limite}'"

        if data_inicio_limite:
            sql_statement += f" AND data_limite >= '{data_inicio_limite}'"

        if data_fim_limite:
            sql_statement += f" AND data_limite <= '{data_fim_limite}'"

        if data_criacao:
            sql_statement += f" AND data_criacao = '{data_criacao}'"

        if data_inicio_criacao:
            sql_statement += f" AND data_criacao >= '{data_inicio_criacao}'"

        if data_fim_criacao:
            sql_statement += f" AND data_criacao <= '{data_fim_criacao}'"

        sql_statement += ";"

        cur.execute(sql_statement)
        conn.commit()
        
        return {"status": "ok", "message": "Tarefas com os filtros especificados excluídas!"}
    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass

# Exporta a lista de tools
TAREFAS_TOOLS = [add_tarefa, cancel_tarefas]