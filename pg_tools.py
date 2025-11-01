import os
from dotenv import load_dotenv
import psycopg2
from typing import Optional, List
from langchain.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

def get_conn():
    return psycopg2.connect(os.getenv("SQL_URL"))

# Essa classe garante que o objeto de Python passe todos esses campos
class AddTransactionArgs(BaseModel):
    amount: float = Field(..., description="Valor da transação (use positivo).")
    source_text: str = Field(..., description="Texto original do usuário.")
    occurred_at: Optional[str] = Field(
        default=None,
        description="Timestamp ISO 8601; se ausente, usa NOW() no banco."
    )
    type_id: Optional[int] = Field(default=None, description="ID em transaction_types (1=INCOME, 2=EXPENSES, 3=TRANSFER).")
    type_name: Optional[str] = Field(default=None, description="Nome do tipo: INCOME | EXPENSES | TRANSFER.")
    category_id: Optional[int] = Field(default=None, description="FK de categories (opcional).")
    category_name: Optional[str] = Field(
        default=None,
        description=("Nome da categoria (Em pt-br). Se não souber o id, informe um dos nomes:"
                     "comida, besteira, estudo, férias, transporte, moradia, saúde, lazer, contas, investimentos, presente, outros" 
        )
    )
    description: Optional[str] = Field(default=None, description="Descrição (opcional).")
    payment_method: Optional[str] = Field(default=None, description="Forma de pagamento (opcional).")

class AddTarefaArgs(BaseModel):
    responsavel: str = Field(..., description="Nome ou nome completo da pessoa que irá realizar a tarefa.")
    empresa_id: int = Field(..., description="ID da empresa para filtrar as tarefas.")
    situacao: str = Field(..., description="Situação da tarefa: 'PENDENTE' | 'CONCLUÍDA' | 'CANCELADA'. Se ausente, cadastra como 'PENDENTE'.")
    ingrediente: Optional[str] = Field(default=None, description="Ingrediente que será consultado na tarefa.")
    pedido_id: Optional[int] = Field(default=None, description="ID do pedido associado a tarefa.")
    data_conclusao: Optional[str] = Field(default=None, description="Data que a tarefa foi concluída.")
    relator_id: Optional[str] = Field(default=None, description="ID da pessoa que está atribuindo a tarefa.")
    tipo_tarefa: Optional[str] = Field(default=None, description="Tipo da tarefa que será realizada.")
    data_limite: Optional[str] = Field(default=None, description="Data limite para conclusão da tarefa.")

class QueryTarefaArgs(BaseModel):
    limit: int = Field(..., description="Número máximo de tarefas a serem retornadas.")



# Essa classe garante que o objeto de Python passe todos esses campos
class QueryTransactionsArgs(BaseModel):
    text: Optional[str] = Field(default=None, description="Texto a buscar em source_text ou description (opcional).")
    type_name: Optional[str] = Field(default=None, description="Nome do tipo: INCOME | EXPENSES | TRANSFER.")
    date_local: Optional[str] = Field(default=None, description="Data específica da transação que o usuário deseja consultar.")
    date_from_local: Optional[str] = Field(default=None, description="Data de início do filtro de transações que o usuário deseja consultar.")
    date_to_local: Optional[str] = Field(default=None, description="Data final do filtro de transações que o usuário deseja consultar.")
    limit: int = Field(..., description="Número máximo de transações a serem retornadas.")

class UpdateTransactionArgs(BaseModel):
    id: Optional[int] = Field(
        default=None,
        description="ID da transação a atualizar. Se ausente, será feita uma busca por (match_text + date_local)."
    )
    match_text: Optional[str] = Field(
        default=None,
        description="Texto para localizar transação quando id nÃ£o for informado (busca em source_text/description)."
    )
    date_local: Optional[str] = Field(
        default=None,
        description="Data local (YYYY-MM-DD) em America/Sao_Paulo; usado em conjunto com match_text quando id ausente."
    )
    amount: Optional[float] = Field(default=None, description="Novo valor.")
    type_id: Optional[int] = Field(default=None, description="Novo type_id (1/2/3).")
    type_name: Optional[str] = Field(default=None, description="Novo type_name: INCOME | EXPENSES | TRANSFER.")
    category_id: Optional[int] = Field(default=None, description="Nova categoria (id).")
    category_name: Optional[str] = Field(default=None, description="Nova categoria (nome).")
    description: Optional[str] = Field(default=None, description="Nova descrição.")
    payment_method: Optional[str] = Field(default=None, description="Novo meio de pagamento.")
    occurred_at: Optional[str] = Field(default=None, description="Novo timestamp ISO 8601.")

TYPE_ALIASES = {
    # === INCOME ===
    "INCOME": "INCOME",
    "ENTRADA": "INCOME",
    "RECEITA": "INCOME",
    "GANHO": "INCOME",
    "SALÁRIO": "INCOME",
    "SALARIO": "INCOME",
    "RENDA": "INCOME",
    "CREDIT": "INCOME",
    "CRÉDITO": "INCOME",
    "CREDITO": "INCOME",
    "DEPÓSITO": "INCOME",
    "DEPOSITO": "INCOME",
    "LUCRO": "INCOME",
    "BONUS": "INCOME",
    "BÔNUS": "INCOME",

    # === EXPENSES ===
    "EXPENSES": "EXPENSES",
    "SAÍDA": "EXPENSES",
    "SAIDA": "EXPENSES",
    "DESPESA": "EXPENSES",
    "CUSTO": "EXPENSES",
    "GASTO": "EXPENSES",
    "PAGAMENTO": "EXPENSES",
    "DÉBITO": "EXPENSES",
    "DEBITO": "EXPENSES",
    "DEBIT": "EXPENSES",
    "COMPRA": "EXPENSES",
    "INVESTIMENTO": "EXPENSES",
    "CONTA": "EXPENSES",

    # === TRANSFER ===
    "TRANSFER": "TRANSFER",
    "TRANSFERÊNCIA": "TRANSFER",
    "TRANSFERENCIA": "TRANSFER",
    "ENVIO": "TRANSFER",
    "RECEBIMENTO": "TRANSFER",
    "REMESSA": "TRANSFER",
    "PIX": "TRANSFER",
    "TED": "TRANSFER",
    "DOC": "TRANSFER",
    "MOVIMENTAÇÃO": "TRANSFER",
    "MOVIMENTACAO": "TRANSFER",
    "TRANSAÇÃO": "TRANSFER",
    "TRANSACAO": "TRANSFER",
}

#Garante que o campo type da tabela transactions receba um id válido (1=INCOME, 2=EXPENSES, 3=TRANSFER
def _resolve_type_id(cur, type_id: Optional[int], type_name: Optional[str]) -> Optional[int]:
    if type_name:
        t = type_name.strip().upper()
        if t in TYPE_ALIASES:
            t = TYPE_ALIASES[t]
        cur.execute("SELECT id FROM transaction_types WHERE UPPER(type)=%s LIMIT 1;", (t,))
        row = cur.fetchone()
        return row[0] if row else None
    if type_id:
        return int(type_id)
    return 2

def _get_category_id(cur, category_name: Optional[str]) -> Optional[int]:
    if not category_name:
        return None
    cur.execute(
        "SELECT id FROM categories WHERE LOWER(name)= LOWER(%s) LIMIT 1;", (category_name,)
    )
    row = cur.fetchone()
    return row[0] if row else None

# Tool: add_transaction
@tool("add_transaction", args_schema=AddTransactionArgs)
def add_transaction(
    amount: float,
    source_text: str,
    occurred_at: Optional[str] = None,
    type_id: Optional[int] = None,
    type_name: Optional[str] = None,
    category_id: Optional[int] = None,
    category_name: Optional[str] = None,
    description: Optional[str] = None,
    payment_method: Optional[str] = None,
) -> dict:
    """Insere uma transação financeira no banco de dados PostgreSQL.""" # docstring obrigatório da @tools do langchain (estranho, mas legal né?)
    conn = get_conn()
    cur = conn.cursor()
    try:
        resolved_type_id = _resolve_type_id(cur, type_id, type_name)
        if not resolved_type_id:
            return {"status": "error", "message": "Tipo inválido (use type_id ou type_name: INCOME/EXPENSES/TRANSFER)."}

        if category_id is None:
            category_id = _get_category_id(cur, category_name)

        if occurred_at:
            cur.execute(
                """
                INSERT INTO transactions
                    (amount, type, category_id, description, payment_method, occurred_at, source_text)
                VALUES
                    (%s, %s, %s, %s, %s, %s::timestamptz, %s)
                RETURNING id, occurred_at;
                """,
                (amount, resolved_type_id, category_id, description, payment_method, occurred_at, source_text),
            )
        else:
            cur.execute(
                """
                INSERT INTO transactions
                    (amount, type, category_id, description, payment_method, occurred_at, source_text)
                VALUES
                    (%s, %s, %s, %s, %s, NOW(), %s)
                RETURNING id, occurred_at;
                """,
                (amount, resolved_type_id, category_id, description, payment_method, source_text),
            )

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

@tool("add_tarefa", args_schema=AddTarefaArgs)
def add_tarefa(
    amount: str,
    source_text: str,
    occurred_at: Optional[str] = None,
    type_id: Optional[int] = None,
    type_name: Optional[str] = None,
    category_id: Optional[int] = None,
    category_name: Optional[str] = None,
    description: Optional[str] = None,
    payment_method: Optional[str] = None,
) -> dict:
    """Insere uma transação financeira no banco de dados PostgreSQL.""" # docstring obrigatório da @tools do langchain (estranho, mas legal né?)
    conn = get_conn()
    cur = conn.cursor()
    try:
        resolved_type_id = _resolve_type_id(cur, type_id, type_name)
        if not resolved_type_id:
            return {"status": "error", "message": "Tipo inválido (use type_id ou type_name: INCOME/EXPENSES/TRANSFER)."}

        if category_id is None:
            category_id = _get_category_id(cur, category_name)

        if occurred_at:
            cur.execute(
                """
                INSERT INTO transactions
                    (amount, type, category_id, description, payment_method, occurred_at, source_text)
                VALUES
                    (%s, %s, %s, %s, %s, %s::timestamptz, %s)
                RETURNING id, occurred_at;
                """,
                (amount, resolved_type_id, category_id, description, payment_method, occurred_at, source_text),
            )
        else:
            cur.execute(
                """
                INSERT INTO transactions
                    (amount, type, category_id, description, payment_method, occurred_at, source_text)
                VALUES
                    (%s, %s, %s, %s, %s, NOW(), %s)
                RETURNING id, occurred_at;
                """,
                (amount, resolved_type_id, category_id, description, payment_method, source_text),
            )

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

@tool("query_transactions", args_schema=QueryTransactionsArgs)
def query_transactions(
    text: Optional[str] = None,
    type_name: Optional[str] = None,
    date_local: Optional[str] = None,
    date_from_local: Optional[str] = None,
    date_to_local: Optional[str] = None,
    limit: int = 20,
) -> dict:
    """
    Consulta transações com filtros por texto (source_text/description), tipo e datas locais (America/Sao_Paulo).
    Os dados devem vir na seguinte ordem:
        - Intervalo (date_from_local/date_to_local): ASC (cronológico).
        - Caso contrário: DESC (mais recentes primeiro).
    """
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        sql_query = """
            SELECT t.amount
                 , t.description
                 , tt.type
                 , c.name
                 , t.source_text
                 , t.occurred_at
              FROM transactions t
              LEFT JOIN transaction_types tt on tt.id = t.type
              LEFT JOIN categories c on c.id = t.category_id
             WHERE 1 = 1
        """
        
        if type_name:
            t = type_name.strip().upper()
            if t in TYPE_ALIASES:
                t = TYPE_ALIASES[t]        
                sql_query += f" AND tt.type = '{t}'"

        if text:
            text.replace(' ', '%')
            sql_query += f" AND t.description ILIKE '%{text}%' OR t.source_text ILIKE '%{text}%'"

        if date_local:
            sql_query += f" AND t.occurred_at = '{date_local}'"

        if date_from_local:
            sql_query += f" AND t.occurred_at >= '{date_from_local}'"

        if date_to_local:
            sql_query += f" AND t.occurred_at <= '{date_to_local}'"

        if date_from_local or date_to_local:
            sql_query += f" ORDER BY t.occurred_at ASC"
        else:
            sql_query += f" ORDER BY t.occurred_at DESC"

        sql_query += f" LIMIT {limit}"

        cur.execute(sql_query)
        rows = cur.fetchall()
        
        return {"status": "ok", "result_list": rows}
    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass

@tool("total_balance")
def total_balance() -> dict:
    """
    Retorna o saldo ( INCOME - EXPENSES ) em todo o histórico (ignora TRANFER).
    """
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
        """
        SELECT SUM(CASE WHEN tt.type = 'INCOME' THEN amount ELSE amount*-1 END) as total_amount
          FROM transactions t
          LEFT JOIN transaction_types tt ON tt.id = t.type
         WHERE tt.type <> 'TRANSFER';
        """
        )
        total_amount = cur.fetchone()[0]
        if total_amount:
            return {"status": "ok", "total_amount": total_amount}
        else:
            return {"status": "error", "message": "Não foi possível calcular o saldo total"}

    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass        

@tool("daily_balance")
def daily_balance(date_local: str) -> dict:
    """
    Retorna o saldo ( INCOME - EXPENSES ) do dia local informado (YYYY-MM-DD) em America/Sao_Paulo.
    Ignora TRANSFER (type=3)
    """
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
        """
        SELECT SUM(CASE WHEN tt.type = 'INCOME' THEN amount ELSE amount*-1 END) as total_amount
          FROM transactions t
          LEFT JOIN transaction_types tt ON tt.id = t.type
         WHERE tt.type <> 'TRANSFER'
           AND CAST(t.occurred_at AS DATE) = CAST(%s AS DATE);;
        """, (date_local, ))
        total_amount = cur.fetchone()[0]
        if total_amount:
            return {"status": "ok", "total_amount": total_amount}
        else:
            return {"status": "error", "message": "Não foi possível calcular o saldo"}

    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass       

def _local_date_filter_sql(field: str = "occurred_at") -> str:
    """
    Retorna um trecho SQL para filtragem por dia local em America/Sao_Paulo.
    Ex.: (occurred_at AT TIME ZONE 'America/Sao_Paulo')::date = %s::date
    """
    return f"(({field} AT TIME ZONE 'America/Sao_Paulo')::date = %s::date)"

@tool("update_transaction", args_schema=UpdateTransactionArgs)
def update_transaction(
    id: Optional[int] = None,
    match_text: Optional[str] = None,
    date_local: Optional[str] = None,
    amount: Optional[float] = None,
    type_id: Optional[int] = None,
    type_name: Optional[str] = None,
    category_id: Optional[int] = None,
    category_name: Optional[str] = None,
    description: Optional[str] = None,
    payment_method: Optional[str] = None,
    occurred_at: Optional[str] = None,
) -> dict:
    """
    Atualiza uma transação existente.
    Estratégias:
      - Se 'id' for informado: atualiza diretamente por ID.
      - Caso contrário: localiza a transação mais recente que combine (match_text em source_text/description)
        E (date_local em America/Sao_Paulo), entÃ£o atualiza.
    Retorna: status, rows_affected, id, e o registro atualizado.
    """
    if not any([amount, type_id, type_name, category_id, category_name, description, payment_method, occurred_at]):
        return {"status": "error", "message": "Nada para atualizar: forneÃ§a pelo menos um campo (amount, type, category, description, payment_method, occurred_at)."}

    conn = get_conn()
    cur = conn.cursor()
    try:
        # Resolve target_id
        target_id = id
        if target_id is None:
            if not match_text or not date_local:
                return {"status": "error", "message": "Sem 'id': informe match_text e date_local para localizar o registro."}

            match_text.replace(" ", "%")
            # Buscar o mais recente no dia local informado que combine o texto
            cur.execute(
                f"""
                SELECT t.id
                FROM transactions t
                WHERE (t.source_text ILIKE %s OR t.description ILIKE %s)
                  AND {_local_date_filter_sql("t.occurred_at")}
                ORDER BY t.occurred_at DESC
                LIMIT 1;
                """,
                (f"%{match_text}%", f"%{match_text}%", date_local)
            )
            row = cur.fetchone()
            if not row:
                return {"status": "error", "message": "Nenhuma transação encontrada para os filtros fornecidos."}
            target_id = row[0]

        # Resolver type_id / category_id a partir de nomes, se fornecidos
        resolved_type_id = _resolve_type_id(cur, type_id, type_name) if (type_id or type_name) else None
        resolved_category_id = category_id
        if category_name and not category_id:
            resolved_category_id = _get_category_id(cur, category_name)

        # Montar SET dinÃ¢mico
        sets = []
        params: List[object] = []
        if amount is not None:
            sets.append("amount = %s")
            params.append(amount)
        if resolved_type_id is not None:
            sets.append("type = %s")
            params.append(resolved_type_id)
        if resolved_category_id is not None:
            sets.append("category_id = %s")
            params.append(resolved_category_id)
        if description is not None:
            sets.append("description = %s")
            params.append(description)
        if payment_method is not None:
            sets.append("payment_method = %s")
            params.append(payment_method)
        if occurred_at is not None:
            sets.append("occurred_at = %s::timestamptz")
            params.append(occurred_at)

        if not sets:
            return {"status": "error", "message": "Nenhum campo válido para atualizar."}

        params.append(target_id)

        cur.execute(
            f"UPDATE transactions SET {', '.join(sets)} WHERE id = %s;",
            params
        )
        rows_affected = cur.rowcount
        conn.commit()

        # Retornar o registro atualizado
        cur.execute(
            """
            SELECT
              t.id, t.occurred_at, t.amount, tt.type AS type_name,
              c.name AS category_name, t.description, t.payment_method, t.source_text
            FROM transactions t
            JOIN transaction_types tt ON tt.id = t.type
            LEFT JOIN categories c ON c.id = t.category_id
            WHERE t.id = %s;
            """,
            (target_id,)
        )
        r = cur.fetchone()
        updated = None
        if r:
            updated = {
                "id": r[0],
                "occurred_at": str(r[1]),
                "amount": float(r[2]),
                "type": r[3],
                "category": r[4],
                "description": r[5],
                "payment_method": r[6],
                "source_text": r[7],
            }

        return {
            "status": "ok",
            "rows_affected": rows_affected,
            "id": target_id,
            "updated": updated
        }

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
TOOLS = [add_transaction, total_balance, daily_balance, query_transactions]
TAREFAS_TOOLS = [add_transaction]