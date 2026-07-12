"""Milvus + MySQL connectivity test (no emoji for Windows)"""
import asyncio
from backend.config import get_settings


async def main():
    s = get_settings()
    print("=" * 55)
    print("  Milvus + MySQL Connectivity Test")
    print("=" * 55)

    # ── MySQL ──
    print(f"\n[MySQL] {s.db_host}:{s.db_port}/{s.db_name} ...")
    try:
        import aiomysql
        conn = await aiomysql.connect(
            host=s.db_host, port=s.db_port,
            user=s.db_user, password=s.db_password,
            db=s.db_name, connect_timeout=5,
            charset="utf8mb4",
        )
        async with conn.cursor() as cur:
            await cur.execute("SELECT VERSION()")
            ver = (await cur.fetchone())[0]
            await cur.execute("SHOW TABLES")
            tables = [t[0] for t in (await cur.fetchall())]
        conn.close()
        print(f"  [OK] Connected! MySQL {ver}")
        print(f"  Tables ({len(tables)}): {tables}")
    except ImportError:
        print("  [FAIL] aiomysql not installed: pip install aiomysql")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # ── Milvus ──
    print(f"\n[Milvus] {s.milvus_host}:{s.milvus_port} ...")
    try:
        from pymilvus import connections, utility, Collection
        connections.connect(host=s.milvus_host, port=s.milvus_port, timeout=8)
        cols = utility.list_collections()
        print(f"  [OK] Connected!")
        print(f"  Collections: {cols}")

        cn = s.milvus_collection_name
        if cn in cols:
            col = Collection(cn)
            col.load()
            print(f"  Collection '{cn}': {col.num_entities} entities")
        else:
            print(f"  Collection '{cn}' not yet created (auto-create on first run)")
        connections.disconnect("default")
    except ImportError:
        print("  [FAIL] pymilvus not installed: pip install pymilvus")
    except Exception as e:
        print(f"  [FAIL] {e}")

    print("\n" + "=" * 55)
    print("Test complete")


asyncio.run(main())
