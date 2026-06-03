import re
from typing import List, Dict

# Telecom Glossary Expansion
TELECOM_GLOSSARY = {
    "ho": "handover",
    "rrc": "radio resource control",
    "ue": "user equipment",
    "enb": "eNodeB",
    "gnb": "gNodeB",
    "nr": "new radio",
    "lte": "long term evolution",
    "pdcp": "packet data convergence protocol",
    "rlc": "radio link control",
    "mac": "medium access control",
    "phy": "physical layer",
    "harq": "hybrid automatic repeat request",
    "drx": "discontinuous reception",
    "qos": "quality of service",
    "kpi": "key performance indicator",
    "rsrp": "reference signal received power",
    "rsrq": "reference signal received quality",
    "sinr": "signal to interference plus noise ratio",
    "prb": "physical resource block",
    "mib": "master information block",
    "sib": "system information block",
    "cqi": "channel quality indicator",
    "pmi": "precoding matrix indicator",
    "ri": "rank indicator",
    "csi": "channel state information",
    "bwp": "bandwidth part",
    "ssb": "synchronization signal block",
    "rach": "random access channel",
    "msg1": "message 1",
    "msg2": "message 2",
    "msg3": "message 3",
    "msg4": "message 4",
    "pusch": "physical uplink shared channel",
    "pucch": "physical uplink control channel",
    "pdsch": "physical downlink shared channel",
    "pdcch": "physical downlink control channel",
    "dci": "downlink control information",
    "uci": "uplink control information",
    "srs": "sounding reference signal",
    "ptr": "phase tracking reference signal",
    "dmrs": "demodulation reference signal",
    "csirs": "channel state information reference signal",
    "amf": "access and mobility management function",
    "smf": "session management function",
    "upf": "user plane function",
    "ausf": "authentication server function",
    "udm": "unified data management",
    "pcf": "policy control function",
    "nrf": "network repository function",
    "nssf": "network slice selection function",
    "nef": "network exposure function",
    "af": "application function",
    "scell": "secondary cell",
    "pcell": "primary cell",
    "spcell": "special cell",
    "pscell": "primary scell",
    "mbsfn": "multicast broadcast single frequency network",
    "ng-ran": "next generation radio access network",
    "e-utran": "evolved universal terrestrial radio access network",
    "epc": "evolved packet core",
    "5gc": "5g core network",
    "cu": "central unit",
    "du": "distributed unit",
    "ru": "radio unit",
    "fdd": "frequency division duplex",
    "tdd": "time division duplex",
    "sa": "standalone",
    "nsa": "non-standalone",
    "en-dc": "e-utra-nr dual connectivity",
    "cg": "cell group",
    "mcg": "master cell group",
    "scg": "secondary cell group",
}

class QueryProcessor:
    def __init__(self):
        self.glossary = TELECOM_GLOSSARY

    def expand_query(self, query: str) -> str:
        """Expands telecom acronyms in the query based on the glossary."""
        # Add word boundaries so we don't replace parts of words (e.g. 'the' -> 't+handover+e' if 'ho' was in it)
        expanded_query = query
        for term, expansion in self.glossary.items():
            # regex with word boundaries \b, case insensitive
            pattern = re.compile(rf"\b{term}\b", re.IGNORECASE)
            expanded_query = pattern.sub(f"{term} ({expansion})", expanded_query)
        return expanded_query

    def classify_query(self, query: str) -> str:
        """Classifies the query into one of 5 types."""
        q_lower = query.lower()
        
        # 1. Definition
        if any(kw in q_lower for kw in ["what is", "what are", "define", "definition", "meaning", "stands for"]):
            return "definition"
            
        # 2. Comparison
        if any(kw in q_lower for kw in ["vs", "versus", "difference", "compare", "better than", "advantages", "disadvantages"]):
            return "comparison"
            
        # 3. Troubleshooting
        if any(kw in q_lower for kw in ["fail", "error", "drop", "issue", "problem", "degrade", "why did", "disconnect", "cause"]):
            return "troubleshooting"
            
        # 4. Optimization
        if any(kw in q_lower for kw in ["improve", "optimize", "increase", "throughput", "latency", "better", "maximize", "minimize", "reduce"]):
            return "optimization"
            
        # 5. Standards QA (default)
        return "standards_qa"

    def decompose_query(self, query: str) -> List[str]:
        """Splits multi-part questions into individual sub-queries."""
        # Simple heuristic: split by "?" if multiple questions exist, or split by " and "
        parts = []
        if "?" in query:
            split_q = query.split("?")
            for p in split_q:
                p = p.strip()
                if len(p) > 5:  # ensure it's not empty or too short
                    parts.append(p + "?")
        else:
            parts = [query]
            
        # Further split by " and " if applicable, but only if they are distinct questions
        final_parts = []
        for p in parts:
            if " and what is " in p.lower() or " and how " in p.lower():
                sub_parts = re.split(r"(?i)\s+and\s+(?=what is|how)", p)
                final_parts.extend([sp.strip() for sp in sub_parts])
            else:
                final_parts.append(p)
                
        return final_parts

    def process(self, query: str) -> Dict:
        """Runs the full pipeline on a query."""
        expanded = self.expand_query(query)
        q_type = self.classify_query(query)
        decomposed = self.decompose_query(expanded)
        
        return {
            "original_query": query,
            "expanded_query": expanded,
            "query_type": q_type,
            "sub_queries": decomposed
        }

if __name__ == "__main__":
    processor = QueryProcessor()
    
    test_queries = [
        "What is RRC?",
        "Why did the HO fail during NG-RAN handover?",
        "How to improve PDCP throughput?",
        "Compare FDD vs TDD in 5G",
        "Explain MAC HARQ process and what is DRX?"
    ]
    
    print("=== Query Understanding Tests ===")
    for q in test_queries:
        print(f"\n[Original] {q}")
        res = processor.process(q)
        print(f"  Type:      {res['query_type']}")
        print(f"  Expanded:  {res['expanded_query']}")
        if len(res['sub_queries']) > 1:
            print(f"  Decomposed: {res['sub_queries']}")
