#!/usr/bin/env python3
"""
RAG Pipeline fsspec test - demonstrates secure RAG with AltaStata

This test:
1. Uploads sample documents to encrypted AltaStata storage
2. Loads documents using LangChain with fsspec integration
3. Creates embeddings and vector store
4. Performs semantic search queries
5. Cleans up test data

Requirements:
    pip install altastata fsspec langchain langchain-community sentence-transformers faiss-cpu
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from altastata.altastata_functions import AltaStataFunctions
from altastata.fsspec import create_filesystem
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def create_sample_documents():
    """Create sample policy documents for testing"""
    documents = {
        "company_policy.txt": """Company Data Retention Policy

Effective Date: January 1, 2024

Overview:
This policy establishes guidelines for the retention and disposal of company data to ensure compliance with legal requirements and operational needs.

Data Retention Periods:
- Employee Records: 7 years after employment termination
- Financial Records: 10 years from the end of the fiscal year
- Customer Data: 5 years after last transaction
- Email Communications: 3 years from date of creation
- Project Documentation: Permanent retention for active projects, 5 years after project completion

Security Requirements:
All retained data must be encrypted at rest and in transit. Access to sensitive data is restricted to authorized personnel only. Data destruction must follow secure deletion procedures.

Compliance:
This policy complies with GDPR, HIPAA, and SOC 2 requirements. Regular audits are conducted quarterly to ensure adherence.

Contact:
For questions regarding data retention, contact the Data Governance team at datagovernance@company.com
""",
        
        "security_guidelines.txt": """Enterprise Security Guidelines

Version 2.0 - Updated March 2024

Purpose:
These guidelines establish security best practices for all employees to protect company data and systems.

Password Requirements:
- Minimum 12 characters
- Mix of uppercase, lowercase, numbers, and special characters
- Change passwords every 90 days
- No password reuse for last 10 passwords
- Enable multi-factor authentication (MFA) for all accounts

Data Classification:
- Public: Information freely shareable
- Internal: For company use only
- Confidential: Restricted to specific teams
- Highly Confidential: Requires executive approval for access

Encryption Standards:
- AES-256 for data at rest
- TLS 1.3 for data in transit
- End-to-end encryption for sensitive communications
- Hardware security modules (HSM) for key management

Access Control:
- Principle of least privilege applies to all systems
- Role-based access control (RBAC) enforced
- Regular access reviews conducted quarterly
- Immediate revocation upon employment termination

Incident Response:
Report security incidents immediately to security@company.com or call the security hotline at 1-800-SEC-HELP.

Training:
All employees must complete annual security awareness training. New employees must complete training within 30 days of hire.
""",

        "remote_work_policy.txt": """Remote Work Policy

Effective: March 2024

Eligibility:
All full-time employees are eligible for remote work arrangements with manager approval.

Work Schedule:
- Core hours: 10 AM - 3 PM in your local timezone
- Flexible hours outside core time
- Required attendance at weekly team meetings
- Availability during business hours via Slack/Teams

Equipment & Technology:
- Company-provided laptop with encrypted hard drive
- VPN access mandatory for all remote connections
- Secure home WiFi network required
- Use of company-approved collaboration tools only

Security Requirements:
- Lock screen when away from workstation
- No public WiFi for accessing company systems
- Secure storage of company documents
- Report lost or stolen equipment immediately

Workspace Requirements:
- Dedicated workspace with minimal distractions
- Ergonomic setup recommended
- Professional background for video calls
- Adequate lighting and internet connectivity

Performance Expectations:
- Same productivity standards as in-office work
- Regular check-ins with manager
- Timely response to communications
- Participation in virtual team activities
""",

        "ai_usage_policy.txt": """Artificial Intelligence Usage Policy

Version 1.0 - April 2024

Purpose:
This policy governs the use of AI tools and services to ensure responsible, ethical, and secure usage across the organization.

Approved AI Tools:
- GitHub Copilot (for software development)
- ChatGPT Enterprise (with business associate agreement)
- Grammarly Business (for writing assistance)
- Custom internal AI tools approved by IT

Prohibited Activities:
- Sharing confidential or proprietary data with public AI services
- Using AI to generate content without human review
- Bypassing security controls to access unapproved AI tools
- Using AI for automated decision-making without human oversight

Data Protection:
- Never input customer PII into AI tools
- Use only anonymized or synthetic data for AI training
- Review all AI-generated content for accuracy and bias
- Maintain audit logs of AI tool usage

Compliance:
- All AI usage must comply with GDPR, CCPA, and industry regulations
- Regular assessments of AI tools for security and privacy risks
- Vendor agreements must include data processing addendums
- Quarterly reviews of AI usage patterns and risks

Training:
All employees using AI tools must complete the AI Ethics and Security training module.

Questions:
Contact the AI Governance team at ai-governance@company.com
"""
    }
    
    return documents


def test_rag_pipeline():
    """Test complete RAG pipeline with AltaStata"""
    print("🚀 Testing RAG Pipeline with AltaStata fsspec")
    print("=" * 80)
    
    # Initialize AltaStata
    print("\n1️⃣  Initializing AltaStata connection...")
    altastata_functions = AltaStataFunctions.from_account_dir(
        '/Users/sergevilvovsky/.altastata/accounts/amazon.rsa.bob123'
    )
    altastata_functions.set_password("123")
    
    # Create filesystem instance
    fs = create_filesystem(altastata_functions, "bob123")
    
    print("✅ AltaStata initialized")
    
    # Upload sample documents
    print("\n2️⃣  Uploading sample documents to encrypted storage...")
    sample_docs = create_sample_documents()
    test_dir = "RAGTest/documents"
    
    for filename, content in sample_docs.items():
        file_path = f"{test_dir}/{filename}"
        result = altastata_functions.create_file(file_path, content.encode('utf-8'))
        print(f"   ✅ Uploaded: {filename} - {result.getOperationStateValue()}")
    
    # Load documents using fsspec directly
    print("\n3️⃣  Loading documents via fsspec...")
    try:
        # Load each file individually using fsspec
        from langchain_core.documents import Document
        documents = []
        for filename in sample_docs.keys():
            file_path = f"{test_dir}/{filename}"
            try:
                # Read file content using fsspec
                with fs.open(file_path, "r") as f:
                    content = f.read()
                    # Create LangChain document
                    doc = Document(
                        page_content=content,
                        metadata={"source": file_path}
                    )
                    documents.append(doc)
                    print(f"   ✅ Loaded: {filename} ({len(content)} chars)")
            except Exception as e:
                print(f"   ❌ Failed to load {filename}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"✅ Total documents loaded: {len(documents)}")
    
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        raise
    
    # Split documents into chunks
    print("\n4️⃣  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks")
    
    # Create embeddings and vector store
    print("\n5️⃣  Creating embeddings and vector store...")
    print("   (This may take a minute on first run - downloading model)")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print("✅ Vector store created successfully")
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        raise
    
    # Test queries
    print("\n6️⃣  Testing RAG queries...")
    print("=" * 80)
    
    test_queries = [
        "What are the password requirements?",
        "How long do we keep financial records?",
        "What is the remote work policy for equipment?",
        "Which AI tools are approved for use?",
        "What encryption standards are required?"
    ]
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📊 Query {i}: {query}")
        print("-" * 80)
        
        try:
            relevant_docs = retriever.get_relevant_documents(query)
            print(f"Found {len(relevant_docs)} relevant chunks:\n")
            
            for j, doc in enumerate(relevant_docs, 1):
                print(f"[Chunk {j}] (from: {os.path.basename(doc.metadata.get('source', 'Unknown'))})")
                print(f"{doc.page_content[:200]}...")
                print()
        
        except Exception as e:
            print(f"❌ Error querying: {e}")
    
    # Test similarity search with scores
    print("\n7️⃣  Testing similarity search with scores...")
    print("=" * 80)
    query = "What are the data retention policies?"
    print(f"\nQuery: {query}\n")
    
    try:
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            print(f"[Result {i}] Similarity: {score:.4f}")
            print(f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}")
            print(f"Content: {doc.page_content[:150]}...")
            print()
    except Exception as e:
        print(f"❌ Error in similarity search: {e}")
    
    # Cleanup
    print("\n8️⃣  Cleaning up test data...")
    try:
        # Delete all test documents
        for filename in sample_docs.keys():
            file_path = f"{test_dir}/{filename}"
            result = altastata_functions.delete_files(file_path, False, None, None)
            print(f"   ✅ Deleted: {filename}")
        
        # Try to delete the directory (it's okay if this fails)
        try:
            altastata_functions.delete_files(test_dir, True, None, None)
            print(f"   ✅ Deleted directory: {test_dir}")
        except:
            pass  # Directory might not be empty or might not support deletion
            
    except Exception as e:
        print(f"⚠️  Warning during cleanup: {e}")
    
    print("\n" + "=" * 80)
    print("✅ RAG Pipeline Test Completed Successfully!")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✅ Encrypted document storage with AltaStata")
    print("  ✅ LangChain integration via fsspec")
    print("  ✅ Semantic search with embeddings")
    print("  ✅ Document chunking and retrieval")
    print("  ✅ Similarity scoring")
    print("\nNext Steps:")
    print("  • Integrate with your preferred LLM (OpenAI, Anthropic, etc.)")
    print("  • Add RetrievalQA chain for automated Q&A")
    print("  • Implement conversation memory for multi-turn dialogues")
    print("  • Deploy with Confidential Computing for maximum security")


if __name__ == "__main__":
    try:
        test_rag_pipeline()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

