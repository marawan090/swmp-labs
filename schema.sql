-- ============================================================================
-- SWMP Labs Academic Research Assistant — Database Schema (Supabase / PostgreSQL)
-- Includes Country-Based and University/Institution-Based Research Filtering Schema Support
-- ============================================================================

-- 1. Saved Papers / Bookmarks
-- Updated to include both `country` and `institution` columns for multi-filter literature management.
CREATE TABLE IF NOT EXISTS saved_papers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    paper_id VARCHAR(256) NOT NULL,
    country VARCHAR(100),                          -- e.g. 'Canada', 'USA', 'Germany', 'United Kingdom'
    institution VARCHAR(200),                      -- e.g. 'Stanford University', 'University of Waterloo', 'MIT'
    paper_metadata JSONB DEFAULT '{}'::jsonb,     -- Full normalized Paper object including country, countries & institutions
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, paper_id)
);

-- Indices for rapid querying and compositional parametric filtering
CREATE INDEX IF NOT EXISTS idx_saved_papers_user_id ON saved_papers(user_id);
CREATE INDEX IF NOT EXISTS idx_saved_papers_country ON saved_papers(country);
CREATE INDEX IF NOT EXISTS idx_saved_papers_institution ON saved_papers(institution);
CREATE INDEX IF NOT EXISTS idx_saved_papers_paper_id ON saved_papers(paper_id);

-- 2. Research Sessions (History)
CREATE TABLE IF NOT EXISTS research_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    title VARCHAR(200) DEFAULT 'Untitled session',
    search_mode VARCHAR(32) DEFAULT 'topic',       -- 'topic', 'author', 'institution', 'custom'
    query_params JSONB DEFAULT '{}'::jsonb,        -- May include query, author, institution, university, country, limit
    discovered_paper_ids TEXT[] DEFAULT ARRAY[]::TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_research_sessions_user ON research_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_research_sessions_created ON research_sessions(created_at DESC);

-- 3. Session Messages (Dialogue / Thread)
CREATE TABLE IF NOT EXISTS session_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES research_sessions(id) ON DELETE CASCADE,
    role VARCHAR(16) NOT NULL,                     -- 'user', 'assistant'
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_session_messages_session ON session_messages(session_id);

-- 4. Saved Research Artifacts (Outreach emails, thesis sections, IEEE literature reviews)
CREATE TABLE IF NOT EXISTS saved_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    session_id UUID REFERENCES research_sessions(id) ON DELETE SET NULL,
    artifact_type VARCHAR(32) NOT NULL,            -- 'outreach', 'thesis', 'ieee_review'
    title VARCHAR(200) DEFAULT '',
    target_paper_id VARCHAR(128),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_saved_artifacts_user ON saved_artifacts(user_id);
CREATE INDEX IF NOT EXISTS idx_saved_artifacts_type ON saved_artifacts(artifact_type);

-- Row Level Security (RLS) Policies
ALTER TABLE saved_papers ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE saved_artifacts ENABLE ROW LEVEL SECURITY;

-- Standard owner policies:
CREATE POLICY "Users can manage their own saved papers"
    ON saved_papers FOR ALL
    USING (auth.uid() = user_id);

CREATE POLICY "Users can manage their own research sessions"
    ON research_sessions FOR ALL
    USING (auth.uid() = user_id);

CREATE POLICY "Users can manage messages in their own sessions"
    ON session_messages FOR ALL
    USING (EXISTS (
        SELECT 1 FROM research_sessions
        WHERE research_sessions.id = session_messages.session_id
        AND research_sessions.user_id = auth.uid()
    ));

CREATE POLICY "Users can manage their own saved artifacts"
    ON saved_artifacts FOR ALL
    USING (auth.uid() = user_id);
