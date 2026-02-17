#!/bin/bash
# =============================================================================
# Personal AI Framework - Run Script
# =============================================================================
COMMAND=${1:-help}
case $COMMAND in
    start)
        echo "🚀 Starting Personal AI services..."
        docker compose up -d llm-server vectordb
        echo "⏳ Waiting for model to load (30s)..."
        sleep 30
        
        # Health check
        if curl -s http://localhost:8080/health | grep -q "healthy"; then
            echo "✅ LLM Server: Online"
        else
            echo "❌ LLM Server: Not responding"
        fi
        
        echo ""
        echo "Starting web UI on http://localhost:3000"
        cd web && python3 -m http.server 3000 &
        echo "✅ Web UI started"
        echo ""
        echo "Open http://localhost:3000 in your browser"
        ;;
        
    stop)
        echo "🛑 Stopping services..."
        docker compose down
        pkill -f "python3 -m http.server 3000" 2>/dev/null || true
        echo "✅ Services stopped"
        ;;
        
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
        
    status)
        echo "📊 Service Status:"
        echo ""
        docker compose ps
        echo ""
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            HEALTH=$(curl -s http://localhost:8080/health)
            echo "LLM Server: ✅ Online"
            echo "  Documents: $(echo $HEALTH | grep -o '"knowledge_base_documents":[0-9]*' | cut -d: -f2)"
        else
            echo "LLM Server: ❌ Offline"
        fi
        ;;
        
    ingest)
        echo "📚 Ingesting knowledge base..."
        python3 pipeline/ingest_knowledge.py
        ;;
        
    ingest-emails)
        echo "📧 Ingesting PST emails..."
        python3 pipeline/ingest_pst_emails.py "$@"
        ;;
        
    logs)
        docker compose logs -f llm-server
        ;;
        
    shell)
        docker exec -it personal-ai-framework-llm-server-1 /bin/bash
        ;;
        
    clear-knowledge)
        read -p "⚠️  This will delete all ingested knowledge. Are you sure? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            curl -X DELETE http://localhost:8080/knowledge/clear
            rm -f knowledge/.ingest_manifest.json
            rm -f knowledge/emails/.pst_ingest_manifest.json
            echo "✅ Knowledge base cleared"
        fi
        ;;
        
    watch)
        echo "👁️  Starting file watcher..."
        python3 pipeline/sync_service.py watch --debounce ${2:-300}
        ;;
        
    sync-now)
        echo "🔄 Forcing immediate sync..."
        python3 pipeline/sync_service.py full-sync
        ;;

    summarize)
        shift
        echo "🎤 Meeting Summarizer"
        python3 pipeline/meeting_summarizer.py "$@"
        ;;
        
    help|*)
        echo "Personal AI Framework"
        echo ""
        echo "Usage: ./run.sh <command>"
        echo ""
        echo "Commands:"
        echo "  start           Start all services"
        echo "  stop            Stop all services"
        echo "  restart         Restart all services"
        echo "  status          Show service status"
        echo "  ingest          Ingest documents from knowledge/ folder"
        echo "  ingest-emails   Ingest extracted PST emails"
        echo "  logs            View LLM server logs"
        echo "  shell           Open shell in LLM container"
        echo "  clear-knowledge Clear the knowledge base"
        echo "  watch           Start file watcher (auto-sync on changes)"
        echo "  sync-now        Force immediate sync"
        echo "  summarize       Transcribe and summarize a meeting recording"
        echo "  help            Show this help"
        ;;
esac
