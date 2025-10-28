"""
Final test with correct filter syntax
"""

import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ONSPRING_API_KEY")
API_BASE = "https://api.onspring.com"


async def final_test():
    """Test the working filter to get all policies."""
    
    print("=" * 60)
    print("FINAL TEST: Query All Policies from App 74")
    print("=" * 60)
    print()
    
    headers = {
        "X-ApiKey": API_KEY,
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        
        # Test with "ne ''" (not equal to empty string)
        print("TEST: Query with 'ne empty string' filter...")
        print("-" * 60)
        
        payload = {
            "appId": 74,
            "dataFormat": "Formatted",
            "filter": "4785 ne ''"  # Purpose field, not equal to empty string
        }
        
        try:
            url = f"{API_BASE}/Records/Query"
            print(f"POST {url}")
            print(f"Payload: {payload}")
            print()
            
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                total = data.get("totalRecords", 0)
                items = data.get("items", [])
                
                print(f"✅ SUCCESS! Found {total} total records")
                print(f"   Returned {len(items)} records in this page")
                print()
                
                if items:
                    print("Sample records:")
                    print("-" * 60)
                    
                    for i, record in enumerate(items[:3], 1):
                        print(f"\nRecord {i}:")
                        print(f"  Record ID: {record.get('recordId')}")
                        print(f"  App ID: {record.get('appId')}")
                        print(f"  Fields:")
                        
                        for field_data in record.get('fieldData', [])[:5]:
                            field_id = field_data.get('fieldId')
                            value = field_data.get('value')
                            field_type = field_data.get('type', 'Unknown')
                            
                            # Truncate long values
                            if isinstance(value, str) and len(value) > 50:
                                value = value[:50] + "..."
                            
                            print(f"    Field {field_id} ({field_type}): {value}")
                    
                    if len(items) > 3:
                        print(f"\n  ... and {len(items) - 3} more records")
                
                print("\n" + "=" * 60)
                print("🎉 INTEGRATION WORKING!")
                print("=" * 60)
                print("\nYour OnSpring integration is now functional!")
                print(f"✅ Can query App 74 (FMOL - Policies)")
                print(f"✅ Found {total} policy records")
                print(f"✅ Filter syntax: 4785 ne ''")
                print("\nNext steps:")
                print("1. Update your MCP server with the corrected code")
                print("2. Test with your client.py")
                print("3. Ask policy questions and get real OnSpring data!")
                
            else:
                error = response.json()
                print(f"❌ Failed: {error}")
                
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(final_test())