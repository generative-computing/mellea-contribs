"""Tool usage example with Mellea and CrewAI.

This example demonstrates how to use tools with CrewAI agents powered by Mellea.
It shows how to create custom tools and use them with agents to perform tasks
that require external actions or data retrieval.

Requirements:
    - Ollama running locally (or configure a different backend)
    - mellea-crewai installed
    - A model with tool-calling capabilities (e.g., llama3.1, mistral, qwen2.5)

Note:
    Tool calling requires a model that supports function calling. Smaller models
    like granite4:micro-h may not reliably use tools. For best results, use:
    - Ollama: llama3.1, mistral, qwen2.5-coder
    - OpenAI: gpt-4, gpt-3.5-turbo
    - Other providers with tool-calling support

Run:
    python examples/tool_usage_example.py
"""

import json
from datetime import datetime
from typing import Any

from crewai import Agent, Crew, Task
from crewai.tools import tool
from mellea import start_session

from mellea_crewai import MelleaLLM


# Define custom tools using CrewAI's @tool decorator
@tool("search_database")
def search_database(query: str) -> str:
    """Search a mock product database for items matching the query.

    Args:
        query: Search query string

    Returns:
        JSON string with search results
    """
    # Mock database
    products = {
        "laptop": {
            "name": "Professional Laptop",
            "price": 1299.99,
            "stock": 15,
            "specs": "16GB RAM, 512GB SSD, Intel i7",
        },
        "mouse": {
            "name": "Wireless Mouse",
            "price": 29.99,
            "stock": 50,
            "specs": "Bluetooth, Ergonomic design",
        },
        "keyboard": {
            "name": "Mechanical Keyboard",
            "price": 89.99,
            "stock": 25,
            "specs": "RGB backlight, Cherry MX switches",
        },
        "monitor": {
            "name": "4K Monitor",
            "price": 449.99,
            "stock": 8,
            "specs": "27-inch, 4K UHD, IPS panel",
        },
    }

    # Search for matching products
    results = []
    query_lower = query.lower()
    for key, product in products.items():
        if query_lower in key or query_lower in product["name"].lower():
            results.append({"id": key, **product})

    if not results:
        return json.dumps({"message": "No products found", "results": []})

    return json.dumps({"message": f"Found {len(results)} product(s)", "results": results})


@tool("calculate_total")
def calculate_total(items: list[dict[str, Any]]) -> str:
    """Calculate the total price for a list of items with quantities.

    Args:
        items: List of dicts with 'price' and 'quantity' keys

    Returns:
        JSON string with calculation details
    """
    subtotal = sum(item["price"] * item["quantity"] for item in items)
    tax_rate = 0.08  # 8% tax
    tax = subtotal * tax_rate
    total = subtotal + tax

    return json.dumps(
        {
            "subtotal": round(subtotal, 2),
            "tax": round(tax, 2),
            "tax_rate": tax_rate,
            "total": round(total, 2),
            "currency": "USD",
        }
    )


@tool("check_inventory")
def check_inventory(product_id: str) -> str:
    """Check the current inventory level for a product.

    Args:
        product_id: Product identifier

    Returns:
        JSON string with inventory information
    """
    # Mock inventory data
    inventory = {
        "laptop": {"available": 15, "reserved": 3, "incoming": 10},
        "mouse": {"available": 50, "reserved": 5, "incoming": 0},
        "keyboard": {"available": 25, "reserved": 2, "incoming": 20},
        "monitor": {"available": 8, "reserved": 1, "incoming": 5},
    }

    if product_id not in inventory:
        return json.dumps({"error": f"Product '{product_id}' not found"})

    data = inventory[product_id]
    return json.dumps(
        {
            "product_id": product_id,
            "available": data["available"],
            "reserved": data["reserved"],
            "incoming": data["incoming"],
            "total_stock": data["available"] + data["reserved"],
            "status": "in_stock" if data["available"] > 0 else "out_of_stock",
        }
    )


@tool("get_current_time")
def get_current_time() -> str:
    """Get the current date and time.

    Returns:
        Current timestamp as ISO format string
    """
    return datetime.now().isoformat()


def main():
    """Run tool usage example with CrewAI and Mellea."""
    print("=" * 60)
    print("Mellea-CrewAI Tool Usage Example")
    print("=" * 60)

    # Create Mellea session
    print("\n1. Creating Mellea session...")
    m = start_session()
    print("   ✓ Mellea session created")

    # Create MelleaLLM
    print("\n2. Creating MelleaLLM...")
    llm = MelleaLLM(mellea_session=m, temperature=0.7)
    print("   ✓ MelleaLLM created")

    # Define tools list
    print("\n3. Defining tools...")
    tools = [
        search_database,
        calculate_total,
        check_inventory,
        get_current_time,
    ]
    print("   ✓ Tools defined:")
    for t in tools:
        print(f"      - {t.name}: {t.description}")

    # Create a sales assistant agent with tools
    print("\n4. Creating sales assistant agent with tools...")
    sales_assistant = Agent(
        role="Sales Assistant",
        goal="Help customers find products and calculate order totals",
        backstory=(
            "You are a helpful sales assistant with access to the product database. "
            "You can search for products, check inventory levels, and calculate "
            "order totals including tax. Always provide accurate information and "
            "be friendly and professional."
        ),
        tools=tools,
        llm=llm,
        verbose=True,
    )
    print("   ✓ Sales assistant agent created with tools")

    # Create a customer inquiry task
    print("\n5. Creating customer inquiry task...")
    inquiry_task = Task(
        description=(
            "A customer is interested in buying a laptop and a mouse. "
            "Please:\n"
            "1. Search for these products in the database\n"
            "2. Check the inventory levels for both items\n"
            "3. Calculate the total cost for 1 laptop and 2 mice (including tax)\n"
            "4. Provide a summary with product details, availability, and total price"
        ),
        agent=sales_assistant,
        expected_output=(
            "A detailed summary including product information, inventory status, "
            "and total cost calculation"
        ),
    )
    print("   ✓ Customer inquiry task created")

    # Create crew and execute
    print("\n6. Creating crew and executing task...")
    crew = Crew(
        agents=[sales_assistant],
        tasks=[inquiry_task],
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Executing Crew with Tools...")
    print("=" * 60 + "\n")

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    print(result)
    print("=" * 60)

    # Show token usage
    print("\n7. Token usage:")
    usage = llm.get_token_usage_summary()
    print(f"   Total tokens: {usage.total_tokens}")
    print(f"   Prompt tokens: {usage.prompt_tokens}")
    print(f"   Completion tokens: {usage.completion_tokens}")
    print(f"   Successful requests: {usage.successful_requests}")

    print("\n✓ Example completed successfully!")
    print("\nKey Takeaways:")
    print("  • Tools extend agent capabilities with external actions")
    print("  • CrewAI tools are automatically converted to Mellea format")
    print("  • Agents can use multiple tools to complete complex tasks")
    print("  • Tool calling requires a model with function-calling support")

    print("\nNote:")
    print("  If the agent didn't use tools, try a model with better tool-calling")
    print("  capabilities like llama3.1, mistral, or qwen2.5-coder")


if __name__ == "__main__":
    main()

# Made with Bob
